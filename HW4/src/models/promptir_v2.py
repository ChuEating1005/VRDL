"""PromptIR v2 — drop-in variant with optional SimpleGate FFN and AdaIR-style
frequency-decomposed Prompt Block.

Reuses the unchanged building blocks (LayerNorm, MDTA attention, patch embed,
down/upsample) from the official PromptIR file. Only the FFN and PromptGenBlock
are swappable. Forward graph is identical to PromptIR so the prompt channel
budget (64/128/320) and decoder skip-connection shapes are preserved.

Knobs:
- ffn: "gdfn" (original) | "simplegate" (NAFNet-style, GELU(x1)*x2 -> x1*x2)
- prompt: "base" (official) | "adair" (FFT low/high-freq dual-bank)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .promptir import (
    Attention,
    Downsample,
    LayerNorm,
    OverlapPatchEmbed,
    PromptGenBlock,
    Upsample,
    FeedForward as GDFN,
)


# ---------------------------------------------------------------------------
# SimpleGate FFN (NAFNet-style). Same shapes as GDFN, GELU removed.
# ---------------------------------------------------------------------------
class SimpleGateFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = x1 * x2  # SimpleGate: no activation
        return self.project_out(x)


def _make_ffn(kind: str, dim: int, factor: float, bias: bool) -> nn.Module:
    if kind == "gdfn":
        return GDFN(dim, factor, bias)
    if kind == "simplegate":
        return SimpleGateFFN(dim, factor, bias)
    raise ValueError(f"unknown ffn: {kind}")


# ---------------------------------------------------------------------------
# AdaIR-style Prompt Generation: FFT decomposes the input feature into a low-
# frequency and high-frequency component, each routed through its own learned
# prompt bank. Rain/snow have distinct spectral fingerprints (snow has more
# low-freq structure, rain streaks are high-freq), so per-band routing should
# help the model disentangle them. Output channel count matches PromptGenBlock
# so it is a drop-in replacement.
# ---------------------------------------------------------------------------
class AdaIRPromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, lpf_ratio=0.25):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_size = prompt_size
        self.lpf_ratio = lpf_ratio
        # two banks: low- and high-frequency prompts
        self.prompt_low = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.prompt_high = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.route_low = nn.Linear(lin_dim, prompt_len)
        self.route_high = nn.Linear(lin_dim, prompt_len)
        # learned scalar mixing the two prompts (per channel)
        self.mix = nn.Parameter(torch.zeros(1, prompt_dim, 1, 1))
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def _split_freq(x: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Centered low-pass / high-pass split via rFFT2.

        ratio in (0,1]: fraction of frequency radius kept for low-pass.
        """
        orig_dtype = x.dtype
        x32 = x.float()
        B, C, H, W = x32.shape
        f = torch.fft.rfft2(x32, norm="ortho")  # [B,C,H,W//2+1]
        Hf, Wf = f.shape[-2:]
        ky = torch.fft.fftfreq(H, device=x.device).abs()
        kx = torch.arange(Wf, device=x.device).float() / max(1, W)
        gy, gx = torch.meshgrid(ky, kx, indexing="ij")
        radius = torch.sqrt(gy ** 2 + gx ** 2)
        mask = (radius <= 0.5 * ratio).to(f.dtype)
        f_low = f * mask
        f_high = f - f_low
        x_low = torch.fft.irfft2(f_low, s=(H, W), norm="ortho").to(orig_dtype)
        x_high = torch.fft.irfft2(f_high, s=(H, W), norm="ortho").to(orig_dtype)
        return x_low, x_high

    def _route(self, feat: torch.Tensor, bank: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        B = feat.size(0)
        emb = feat.mean(dim=(-2, -1))
        w = F.softmax(linear(emb), dim=1)  # [B,N]
        # bank: [1,N,C,Hp,Wp]
        prompt = (w[:, :, None, None, None] * bank).sum(dim=1)  # [B,C,Hp,Wp]
        return prompt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_low, x_high = self._split_freq(x, self.lpf_ratio)
        p_low = self._route(x_low, self.prompt_low, self.route_low)
        p_high = self._route(x_high, self.prompt_high, self.route_high)
        gate = torch.sigmoid(self.mix)
        prompt = gate * p_low + (1.0 - gate) * p_high
        prompt = F.interpolate(prompt, size=(H, W), mode="bilinear", align_corners=False)
        prompt = self.conv3x3(prompt)
        return prompt


def _make_prompt(kind: str, prompt_dim: int, prompt_len: int, prompt_size: int, lin_dim: int) -> nn.Module:
    if kind == "base":
        return PromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)
    if kind == "adair":
        return AdaIRPromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)
    raise ValueError(f"unknown prompt: {kind}")


# ---------------------------------------------------------------------------
# Transformer block parameterized by FFN type.
# ---------------------------------------------------------------------------
class TransformerBlockV2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, ffn_kind="gdfn"):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = _make_ffn(ffn_kind, dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# PromptIR v2 — mirrors PromptIR forward graph exactly.
# ---------------------------------------------------------------------------
class PromptIRv2(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple[int, int, int, int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        decoder: bool = True,
        ffn: str = "simplegate",
        prompt: str = "adair",
    ):
        super().__init__()
        self.decoder = decoder
        self.ffn_kind = ffn
        self.prompt_kind = prompt

        def tb(d, h, n):
            return nn.Sequential(*[
                TransformerBlockV2(d, h, ffn_expansion_factor, bias, LayerNorm_type, ffn) for _ in range(n)
            ])

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        if decoder:
            self.prompt1 = _make_prompt(prompt, prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = _make_prompt(prompt, prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = _make_prompt(prompt, prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)

        self.chnl_reduce1 = nn.Conv2d(64, 64, 1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, 1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, 1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, 1, bias=bias)
        self.encoder_level1 = tb(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2) + 128, int(dim * 2), 1, bias=bias)
        self.encoder_level2 = tb(int(dim * 2), heads[1], num_blocks[1])
        self.down2_3 = Downsample(int(dim * 2))

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 4) + 256, int(dim * 4), 1, bias=bias)
        self.encoder_level3 = tb(int(dim * 4), heads[2], num_blocks[2])
        self.down3_4 = Downsample(int(dim * 4))

        self.latent = tb(int(dim * 8), heads[3], num_blocks[3])

        self.up4_3 = Upsample(int(dim * 4))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2) + 192, int(dim * 4), 1, bias=bias)
        self.noise_level3 = TransformerBlockV2(int(dim * 4) + 512, heads[2], ffn_expansion_factor, bias, LayerNorm_type, ffn)
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 4) + 512, int(dim * 4), 1, bias=bias)
        self.decoder_level3 = tb(int(dim * 4), heads[2], num_blocks[2])

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.noise_level2 = TransformerBlockV2(int(dim * 2) + 224, heads[2], ffn_expansion_factor, bias, LayerNorm_type, ffn)
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2) + 224, int(dim * 4), 1, bias=bias)
        self.decoder_level2 = tb(int(dim * 2), heads[1], num_blocks[1])

        self.up2_1 = Upsample(int(dim * 2))
        self.noise_level1 = TransformerBlockV2(int(dim * 2) + 64, heads[2], ffn_expansion_factor, bias, LayerNorm_type, ffn)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2) + 64, int(dim * 2), 1, bias=bias)
        self.decoder_level1 = tb(int(dim * 2), heads[0], num_blocks[0])

        self.refinement = tb(int(dim * 2), heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1
