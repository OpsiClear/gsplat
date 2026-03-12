# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import importlib.util
import os
import os.path as osp
import pathlib
import platform
import shutil
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"


def _is_absolute_manifest_entry(entry: str) -> bool:
    normalized = entry.strip().replace("\\", "/")
    return os.path.isabs(entry.strip()) or (
        len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/"
    )


def purge_stale_egg_info(project_root: pathlib.Path) -> None:
    egg_info_dir = project_root / "gsplat.egg-info"
    sources_file = egg_info_dir / "SOURCES.txt"
    if not sources_file.exists():
        return

    try:
        lines = sources_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    if any(_is_absolute_manifest_entry(line) for line in lines):
        shutil.rmtree(egg_info_dir, ignore_errors=True)


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    project_root = pathlib.Path(__file__).resolve().parent
    build_module_path = project_root / "gsplat" / "cuda" / "build.py"
    spec = importlib.util.spec_from_file_location("gsplat_build", build_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load build helpers from {build_module_path}")
    build_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_module)

    params = build_module.get_build_parameters()

    def _to_setup_relative(path_str: str) -> str:
        path = pathlib.Path(path_str)
        if not path.is_absolute():
            return path.as_posix()
        return path.relative_to(project_root).as_posix()

    extension = CUDAExtension(
        "gsplat.csrc",
        sources=[_to_setup_relative(path) for path in params.sources],
        include_dirs=params.extra_include_paths,
        extra_compile_args={
            "cxx": params.extra_cflags,
            "nvcc": params.extra_cuda_cflags,
        },
        extra_link_args=params.extra_ldflags,
    )
    return [extension]


purge_stale_egg_info(pathlib.Path(__file__).resolve().parent)


setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.3",
            "pytest-env==0.8.1",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml>=6.0.1",
            "build",
            "twine",
            "cupy",
            "nerfacc>=0.5.3",
            "PLAS @ git+https://github.com/fraunhoferhhi/PLAS.git",
            "imageio>=2.37.2",
            "torchpq>=0.3.0.6",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)
