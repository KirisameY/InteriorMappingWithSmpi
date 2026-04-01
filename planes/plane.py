from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, List, Tuple, Optional, TypeVar
import json
import numpy as np
import open3d as o3d
import copy
import os
from PIL import Image


TTex = TypeVar('TTex')

@dataclass
class PlaneRectangle(Generic[TTex]):
    center: np.ndarray
    normal: np.ndarray
    length_vec: np.ndarray
    width_vec: np.ndarray
    texture: Optional[TTex] = None

def write_as_json(planes: List[PlaneRectangle], save_path: str):
    with open(save_path, "w") as f:
        json.dump([{
            "center": rect.center.tolist(),
            "normal": rect.normal.tolist(),
            "length_vec": rect.length_vec.tolist(),
            "width_vec": rect.width_vec.tolist(),
            "texture": f"plane_{i}.png"
        } for i, rect in enumerate(planes)], f, indent=4)

def read_from_json(json_path: str, texture_loader: callable[[str], TTex]) -> List[PlaneRectangle[TTex]]:
    if not os.path.exists(json_path):
        print(f"未找到 {json_path}")
        return []
    
    with open(json_path, "r") as f:
        rects = json.load(f)

    planes = []
    for rect in rects:
        plane = PlaneRectangle(
            center=np.array(rect["center"]),
            normal=np.array(rect["normal"]),
            length_vec=np.array(rect["length_vec"]),
            width_vec=np.array(rect["width_vec"]),
            texture=texture_loader(os.path.join(os.path.dirname(json_path), rect["texture"]))
        )
        planes.append(plane)
    
    return planes