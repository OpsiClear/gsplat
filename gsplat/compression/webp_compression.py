import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.compression.sort import sort_splats
from gsplat.utils import inverse_log_transform, log_transform


@dataclass
class WebpCompression:
    """Uses quantization and sorting to compress splats into WebP files and uses
    K-means clustering to compress the spherical harmonic coefficents.

    .. warning::
        This class requires the `Pillow <https://pypi.org/project/pillow/>`_,
        `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
        and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

    .. warning::
        This class might throw away a few lowest opacities splats if the number of
        splats is not a square number.

    .. note::
        The splats parameters are expected to be pre-activation values. It expects
        the following fields in the splats dictionary: "means", "scales", "quats",
        "opacities", "sh0", "shN". More fields can be added to the dictionary, but
        they will only be compressed using NPZ compression.

    References:
        - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
        - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.
        lossless (bool, optional): Whether to use lossless WebP compression. Defaults to True.
        quality (int, optional): Quality of WebP compression (0-100). Only used when lossless is False. Defaults to 100.
    """

    use_sort: bool = True
    verbose: bool = True
    lossless: bool = True
    quality: int = 100

    def _get_compress_fn(self, param_name: str) -> Callable:
        compress_fn_map = {
            "means": _compress_webp_16bit,
            "scales": _compress_webp,
            "quats": _compress_webp,
            "opacities": _compress_webp,
            "sh0": _compress_webp,
            "shN": _compress_kmeans,
        }
        if param_name in compress_fn_map:
            return compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(self, param_name: str) -> Callable:
        decompress_fn_map = {
            "means": _decompress_webp_16bit,
            "scales": _decompress_webp,
            "quats": _decompress_webp,
            "opacities": _decompress_webp,
            "sh0": _decompress_webp,
            "shN": _decompress_kmeans,
        }
        if param_name in decompress_fn_map:
            return decompress_fn_map[param_name]
        else:
            return _decompress_npz

    def compress(self, compress_dir: str, splats: Dict[str, Tensor]) -> None:
        """Run compression

        Args:
            compress_dir (str): directory to save compressed files
            splats (Dict[str, Tensor]): Gaussian splats to compress
        """

        # Param-specific preprocessing
        splats["means"] = log_transform(splats["means"])
        splats["quats"] = F.normalize(splats["quats"], dim=-1)

        n_gs = len(splats["means"])
        n_sidelen = int(n_gs**0.5)
        n_crop = n_gs - n_sidelen**2
        if n_crop != 0:
            splats = _crop_n_splats(splats, n_crop)
            print(
                f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
            )

        if self.use_sort:
            splats = sort_splats(splats)

        meta = {}
        for param_name in splats.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                "n_sidelen": n_sidelen,
                "verbose": self.verbose,
                "lossless": self.lossless,
                "quality": self.quality,
            }
            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], **kwargs
            )

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def decompress(self, compress_dir: str) -> Dict[str, Tensor]:
        """Run decompression

        Args:
            compress_dir (str): directory that contains compressed files

        Returns:
            Dict[str, Tensor]: decompressed Gaussian splats
        """
        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        splats = {}
        for param_name, param_meta in meta.items():
            decompress_fn = self._get_decompress_fn(param_name)
            splats[param_name] = decompress_fn(compress_dir, param_name, param_meta)

        # Param-specific postprocessing
        splats["means"] = inverse_log_transform(splats["means"])
        return splats


def _write_image(compress_dir, param_name, img, lossless: bool=True, quality: int=100, verbose: bool = False):
    """
    Compresses the image as webp. Centralized function to change
    image encoding in the future if need be.
    """
    from PIL import Image
    filename = f"{param_name}.webp"
    os.makedirs(compress_dir, exist_ok=True)
    Image.fromarray(img).save(
        os.path.join(compress_dir, filename),
        format="webp",
        lossless=lossless,
        quality=quality if not lossless else 100,
        method=6,
        exact=True
    )
    if verbose:
        print(f"✓ {filename}")
        print(f"Lossless: {lossless}")
        print(f"Quality: {quality}")
    return filename


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    opacities = splats["opacities"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices]
    return splats


def _compress_webp(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 8-bit quantization and lossless WebP compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    if torch.numel(params) == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
            "files": [],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()
    filename = _write_image(
        compress_dir, 
        param_name, 
        img, 
        lossless=kwargs.get("lossless", True),
        quality=kwargs.get("quality", 100),
        verbose=kwargs.get("verbose", False)
    )

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": [filename]
    }
    return meta


def _decompress_webp(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters from WebP file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return params

    img = imageio.imread(os.path.join(compress_dir, meta["files"][0]))
    
    # Determine the expected number of channels from the metadata
    expected_channels = meta['shape'][-1] if len(meta['shape']) > 1 else 1

    # If the saved image was grayscale (1 channel), but read as RGB (3 channels), take one channel.
    if img.ndim == 3 and expected_channels == 1:
        img = img[..., 0]
        
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_webp_16bit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and WebP compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    if torch.numel(params) == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
            "files": [],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF

    verbose = kwargs.get("verbose", False)
    lossless = kwargs.get("lossless", True)
    quality = kwargs.get("quality", 100)
    
    file_l = _write_image(compress_dir, f"{param_name}_l", img_l.astype(np.uint8), lossless=lossless, quality=quality, verbose=verbose)
    file_u = _write_image(compress_dir, f"{param_name}_u", img_u.astype(np.uint8), lossless=lossless, quality=quality, verbose=verbose)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "files": [file_l, file_u],
    }
    return meta


def _decompress_webp_16bit(
    compress_dir: str, param_name: str, meta: Dict[str, Any]
) -> Tensor:
    """Decompress parameters from WebP files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return params

    img_l = imageio.imread(os.path.join(compress_dir, meta["files"][0]))
    img_u = imageio.imread(os.path.join(compress_dir, meta["files"][1]))
    img_u = img_u.astype(np.uint16)
    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_npz(
    compress_dir: str, param_name: str, params: Tensor, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with numpy's NPZ compression."""
    npz_dict = {"arr": params.detach().cpu().numpy()}
    save_fp = os.path.join(compress_dir, f"{param_name}.npz")
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    np.savez_compressed(save_fp, **npz_dict)
    meta = {
        "shape": params.shape,
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters with numpy's NPZ compression."""
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    params = torch.tensor(arr)
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_kmeans(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_sidelen: int,
    quantization: int = 8,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run K-means clustering on parameters and save centroids and labels as images.

    .. warning::
        TorchPQ must installed to use K-means clustering.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters to compress
        n_sidelen (int): image side length
        quantization (int): number of bits in quantization
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Any]: metadata
    """
    try:
        from torchpq.clustering import KMeans
    except:
        raise ImportError(
            "Please install extra dependencies with 'pip install torchpq cupy' to use K-means clustering"
        )

    if torch.numel(params) == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    params = params.reshape(params.shape[0], -1)
    dim = params.shape[1]
    n_clusters = round((len(params) >> 2) / 64) * 64
    n_clusters = min(n_clusters, 2 ** 16)

    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)
    labels = kmeans.fit(params.permute(1, 0).contiguous())
    labels = labels.detach().cpu().numpy()
    centroids = kmeans.centroids.permute(1, 0)

    mins = torch.min(centroids)
    maxs = torch.max(centroids)
    centroids_norm = (centroids - mins) / (maxs - mins)
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = (
        (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)
    )

    # sort centroids for compact atlas layout
    sorted_indices = np.lexsort(centroids_quant.T)
    sorted_indices = sorted_indices.reshape(64, -1).T.reshape(-1)
    sorted_centroids_quant = centroids_quant[sorted_indices]
    inverse = np.argsort(sorted_indices)

    centroids_packed = sorted_centroids_quant.reshape(-1, int(dim * 64 / 3), 3)
    labels = inverse[labels].astype(np.uint16).reshape((n_sidelen, n_sidelen))
    labels_l = labels & 0xFF
    labels_u = (labels >> 8) & 0xFF

    # Combine low and high bytes into single texture: R=labels_l, G=labels_u, B=0
    labels_combined = np.zeros((n_sidelen, n_sidelen, 3), dtype=np.uint8)
    labels_combined[..., 0] = labels_l.astype(np.uint8)
    labels_combined[..., 1] = labels_u.astype(np.uint8)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
        "files": [
            _write_image(compress_dir, f"{param_name}_centroids", centroids_packed, verbose=verbose, lossless=kwargs.get("lossless", True), quality=kwargs.get("quality", 100)),
            _write_image(compress_dir, f"{param_name}_labels", labels_combined, verbose=verbose, lossless=kwargs.get("lossless", True), quality=kwargs.get("quality", 100))
        ]
    }
    return meta


def _decompress_kmeans(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from K-means compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio
    
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return params

    centroids_packed_img = imageio.imread(os.path.join(compress_dir, meta["files"][0]))
    labels_combined_img = imageio.imread(os.path.join(compress_dir, meta["files"][1]))

    # Decompress labels
    labels_l = labels_combined_img[..., 0].astype(np.uint16)
    labels_u = labels_combined_img[..., 1].astype(np.uint16)
    labels = (labels_u << 8) | labels_l
    labels = labels.flatten()

    # Decompress centroids
    dim = np.prod(meta['shape'][1:])
    sorted_centroids_quant = centroids_packed_img.reshape(-1, dim)

    centroids_norm = sorted_centroids_quant / (2 ** meta["quantization"] - 1)
    centroids_norm = torch.from_numpy(centroids_norm.astype(np.float32))
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    centroids = centroids_norm * (maxs - mins) + mins

    params = centroids[labels]
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params
