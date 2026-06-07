#!/usr/bin/env python3
"""
專案內 voxel 標籤 .npz 的統一讀寫：單一陣列鍵 ``voxel``。

讀取時仍相容舊檔（arr_0 / data / labels / 第一個鍵）。
"""

from __future__ import annotations

import os
from typing import BinaryIO, Union

import numpy as np

# 寫入 .npz 時使用的陣列名稱（等同於以前未命名時的 arr_0，但語意明確）
VOXEL_NPZ_KEY = "voxel"

FileLike = Union[str, os.PathLike[str], BinaryIO]


def load_voxel_npz(path: Union[str, os.PathLike[str]]) -> np.ndarray:
    """從 .npz 載入體素標籤陣列；優先 ``voxel``，其次舊鍵名。"""
    path_str = os.fspath(path)
    with np.load(path_str, allow_pickle=False) as z:
        if VOXEL_NPZ_KEY in z:
            return np.asarray(z[VOXEL_NPZ_KEY])
        if "arr_0" in z:
            return np.asarray(z["arr_0"])
        if "data" in z:
            return np.asarray(z["data"])
        if "labels" in z:
            return np.asarray(z["labels"])
        return np.asarray(z[z.files[0]])


def save_voxel_npz(file: FileLike, voxel: np.ndarray, *, compressed: bool = True) -> None:
    """
    寫入 .npz，僅含一個陣列鍵 ``voxel``。

    ``file`` 可為路徑字串、Path，或已開啟的二進位緩衝（如 BytesIO）。
    """
    kw = {VOXEL_NPZ_KEY: voxel}
    if compressed:
        np.savez_compressed(file, **kw)
    else:
        np.savez(file, **kw)
