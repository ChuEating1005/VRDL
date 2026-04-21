import os
from pathlib import Path

CUDA_HOME_OVERRIDE = "/usr/local/cuda-12.4"
os.environ.setdefault("CUDA_HOME", CUDA_HOME_OVERRIDE)
os.environ["PATH"] = f"{CUDA_HOME_OVERRIDE}/bin:{os.environ.get('PATH', '')}"

import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension


cpp_extension.CUDA_HOME = CUDA_HOME_OVERRIDE
BuildExtension = cpp_extension.BuildExtension
CUDAExtension = cpp_extension.CUDAExtension


def get_extensions():
    this_dir = Path(__file__).resolve().parent
    extensions_dir = this_dir / "src"
    sources = [str(path) for path in sorted(extensions_dir.glob("*.cpp"))]
    sources += [str(path) for path in sorted((extensions_dir / "cpu").glob("*.cpp"))]
    sources += [str(path) for path in sorted((extensions_dir / "cuda").glob("*.cu"))]

    extra_compile_args = {
        "cxx": ["-O2"],
        "nvcc": [
            "-O2",
            "-DCUDA_HAS_FP16=1",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_89,code=sm_89",
        ],
    }

    return [
        CUDAExtension(
            name="MultiScaleDeformableAttention",
            sources=sources,
            include_dirs=[str(extensions_dir)],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="MultiScaleDeformableAttention",
    version="0.1.0",
    description="CUDA extension for Multi-Scale Deformable Attention",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
