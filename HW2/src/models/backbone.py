from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from src.models.position_encoding import build_position_encoding
from src.util.misc import NestedTensor


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: list[int],
        return_interm_indices: list[int],
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or not any(layer in name for layer in ["layer2", "layer3", "layer4"]):
                parameter.requires_grad_(False)

        stage_names = ["layer1", "layer2", "layer3", "layer4"]
        return_layers = OrderedDict(
            (stage_names[idx], str(out_idx)) for out_idx, idx in enumerate(return_interm_indices)
        )
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> dict[str, NestedTensor]:
        xs = self.body(tensor_list.tensors)
        out = {}
        mask = tensor_list.mask
        for name, x in xs.items():
            if mask is None:
                resized_mask = None
            else:
                resized_mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, resized_mask)
        return out


class Backbone(BackboneBase):
    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool = False,
        return_interm_indices: list[int] = [1, 2, 3],
        batch_norm: type = FrozenBatchNorm2d,
    ):
        weights_map = {
            "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
            "resnet101": torchvision.models.ResNet101_Weights.IMAGENET1K_V2,
            "resnet152": torchvision.models.ResNet152_Weights.IMAGENET1K_V2,
        }
        channel_map = {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
            "resnet152": [256, 512, 1024, 2048],
        }
        if name not in weights_map or name not in channel_map:
            raise ValueError(f"unsupported backbone {name}")

        backbone = getattr(torchvision.models, name)(
            weights=weights_map[name],
            norm_layer=batch_norm,
            replace_stride_with_dilation=[False, False, dilation],
        )
        num_channels = [channel_map[name][idx] for idx in return_interm_indices]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)
        assert self.body.bn1.running_mean.abs().sum() > 0


class Joiner(nn.Sequential):
    def __init__(self, backbone: Backbone, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor) -> tuple[list[NestedTensor], list[torch.Tensor]]:
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args) -> Joiner:
    position_embedding = build_position_encoding(args.hidden_dim, args.position_embedding)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(
        name=args.backbone,
        train_backbone=train_backbone,
        dilation=args.dilation,
        return_interm_indices=args.return_interm_indices,
    )
    return Joiner(backbone, position_embedding)
