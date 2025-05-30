# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for processing record3d data."""

import json
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import io


def record3d_to_json(images_paths: List[Path], metadata_path: Path, output_dir: Path, indices: np.ndarray) -> int:
    """Converts Record3D's metadata and image paths to a JSON file.

    Args:
        images_paths: list if image paths.
        metadata_path: Path to the Record3D metadata JSON file.
        output_dir: Path to the output directory.
        indices: Indices to sample the metadata_path. Should be the same length as images_paths.

    Returns:
        The number of registered images.
    """

    assert len(images_paths) == len(indices)

    metadata_dict = io.load_from_json(metadata_path)

    poses_data = np.array(metadata_dict["poses"])  # (N, 3, 4)
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    camera_to_worlds = np.concatenate(
        [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
        axis=-1,
    ).astype(np.float32)
    camera_to_worlds = camera_to_worlds[indices]

    homogeneous_coord = np.zeros_like(camera_to_worlds[..., :1, :])
    homogeneous_coord[..., :, 3] = 1
    camera_to_worlds = np.concatenate([camera_to_worlds, homogeneous_coord], -2)

    intrinsics_data = np.array(metadata_dict["perFrameIntrinsicCoeffs"])[indices]

    frames = []
    for i, im_path in enumerate(images_paths):
        fx, fy, cx, cy = intrinsics_data[i]
        c2w = camera_to_worlds[i]
        frame = {
            "file_path": im_path.as_posix(),
            "transform_matrix": c2w.tolist(),
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
        }
        frames.append(frame)

    out = {
        "w": metadata_dict["w"],
        "h": metadata_dict["h"],
        "camera_model": CAMERA_MODELS["perspective"].name,
        "frames": frames,
    }

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)
