# Standard library imports
import math
import os
from copy import deepcopy

# Third-party imports
import cv2
import numpy as np
import torch
import yaml


class Homography:
    def __init__(self):
        pass

    @staticmethod
    def projectCCStoPCS(ccs_coords, K, width, height, d=None, mode="skip", device=None):
        """
        Projects set of points in CCS to PCS.
        ccs_coords: (N x 3) array of points in CCS
        Returns: (N x 2) array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        if device is not None and isinstance(ccs_coords, torch.Tensor):
            # Torch tensor path - simplified projection without distortion
            # Simple pinhole projection
            projected = torch.matmul(ccs_coords, K.T)
            pixel_coords = projected[:, :2] / projected[:, 2:3]

            # Convert to integer pixel coordinates and clip to image bounds
            pixel_coords = pixel_coords.round().long()
            pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, width - 1)
            pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, height - 1)

            # For torch path, return simplified output (no mask for now)
            return pixel_coords, None
        else:
            # Original NumPy implementation
            ccs_coords = np.asarray(ccs_coords, dtype=np.float64)
            ccs_mask = (ccs_coords[:, 2] > 0)  # mask to filter out points in front of camera (ie, possibly visible in image). This is important and is not taken care of by pixel bounding
            ccs_coords = ccs_coords[ccs_mask]
            if ccs_coords.shape[0] == 0 or ccs_coords is None:
                return None, None
            R = np.zeros((3, 1), dtype=np.float64)
            T = np.zeros((3, 1), dtype=np.float64)
            if d is None:
                d = np.zeros((5,), dtype=np.float64)
            image_points, _ = cv2.projectPoints(ccs_coords, R, T, K, d)
            image_points = image_points.reshape(-1, 2).astype(int)
            image_points, pcs_mask = Homography.to_image_fov_bounds(image_points, width, height, mode=mode)
            unified_mask = deepcopy(ccs_mask)
            unified_mask[ccs_mask] = pcs_mask
            return image_points, unified_mask

    @staticmethod
    def general_project_A_to_B(inp, AtoBmat, device=None):
        """
        Project inp from A frame to B
        inp: (N x 3) array of points in A frame
        AtoBmat: (4 x 4) transformation matrix from A to B
        Returns: (N x 3) array of points in B frame
        """
        if device is not None and isinstance(inp, torch.Tensor):
            # Torch tensor path
            ones = torch.ones(inp.shape[0], 1, dtype=inp.dtype, device=inp.device)
            inp_4d = torch.cat([inp, ones], dim=1)
            out_4d = torch.matmul(inp_4d, AtoBmat.T)
            out_3d = out_4d[:, :3] / out_4d[:, 3:4]
            return out_3d
        else:
            # NumPy path (original implementation)
            inp = np.asarray(inp).astype(np.float64)
            inp_4d = Homography.get_homo_from_ordinary(inp)
            out_4d = (AtoBmat @ inp_4d.T).T
            return Homography.get_ordinary_from_homo(out_4d)

    @staticmethod
    def to_image_fov_bounds(pixels, width, height, mode="skip"):
        """
        Returns the pixels coords in image bounds and a mask for indicating which pixels were kept (ie, relevant for skip mode)
        """
        if mode == "skip":
            mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            return pixels[mask], mask
        elif mode == "clip":
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
            return pixels, np.ones(pixels.shape[0], dtype=bool)
        elif mode == "none":
            return pixels, np.ones(pixels.shape[0], dtype=bool)
        else:
            raise ValueError("Unknown fov bounds mode!")

    @staticmethod
    def get_ordinary_from_homo(points_higherD):
        # Scales so that last coord is 1 and then removes last coord
        points_higherD = points_higherD / points_higherD[:, -1].reshape(-1, 1)  # scale by the last coord
        return points_higherD[:, :-1]

    @staticmethod
    def get_homo_from_ordinary(points_lowerD):
        # Append 1 to each point
        ones = np.ones((points_lowerD.shape[0], 1))  # create a column of ones
        return np.hstack([points_lowerD, ones])  # append the ones column to points

    @staticmethod
    def get_std_trans(cx=0, cy=0, cz=0, device=None):
        """
        cx, cy, cz are the coords of O_M wrt O_F when expressed in F
        Multiplication goes like M_coords = T * F_coords
        """
        mat = [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 1, -cz],
            [0, 0, 0, 1]
        ]
        if device is not None:
            return torch.tensor(mat, dtype=torch.float32, device=device)
        return np.array(mat)

    @staticmethod
    def get_std_rot(axis, alpha, device=None):
        """
        axis is either "X", "Y", or "Z" axis of F and alpha is positive acc to right hand thumb rule dirn
        Multiplication goes like M_coords = T * F_coords
        """
        if device is not None:
            cos_alpha = torch.cos(torch.tensor(alpha, device=device))
            sin_alpha = torch.sin(torch.tensor(alpha, device=device))
        else:
            cos_alpha = math.cos(alpha)
            sin_alpha = math.sin(alpha)

        if axis == "X":
            mat = [
                [1, 0, 0, 0],
                [0, cos_alpha, sin_alpha, 0],
                [0, -sin_alpha, cos_alpha, 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Y":
            mat = [
                [cos_alpha, 0, -sin_alpha, 0],
                [0, 1, 0, 0],
                [sin_alpha, 0, cos_alpha, 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Z":
            mat = [
                [cos_alpha, sin_alpha, 0, 0],
                [-sin_alpha, cos_alpha, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        else:
            raise ValueError("Invalid axis!")

        if device is not None:
            return torch.tensor(mat, dtype=torch.float32, device=device)
        return np.array(mat)

    @staticmethod
    def get_rectified_K(K, d, w, h, alpha=0.0):
        """
        K: (3 x 3) camera intrinsic matrix
        d: (5,) distortion coefficients
        w: width of the image
        h: height of the image
        alpha: 0.0 -> crop the image to remove black borders, 1.0 -> preserve all pixels but retains black borders
        Returns: (3 x 3) rectified camera intrinsic matrix and (5,) rectified distortion coefficients
        """
        newK, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), alpha=alpha)
        return newK, roi

    @staticmethod
    def rectifyRawCamImage(cv2_img, K, d, alpha=0.0):
        """
        Rectifies (meaning undistorts for monocular) the raw camera image using the camera intrinsics and distortion coefficients.
        cv2_img: (H x W x 3) array of raw camera image
        K: (3 x 3) camera intrinsic matrix
        d: (5,) distortion coefficients
        alpha: 0.0 -> crop the image to remove black borders, 1.0 -> preserve all pixels but retains black borders
        Returns: (H x W x 3) array of rectified camera image
        """
        h, w = cv2_img.shape[:2]
        newK, roi = Homography.get_rectified_K(K, d, w, h, alpha)  # roi is the region of interest
        map1, map2 = cv2.initUndistortRectifyMap(K, d, None, newK, (w, h), cv2.CV_16SC2)
        undistorted_img = cv2.remap(cv2_img, map1, map2, interpolation=cv2.INTER_LINEAR)
        if alpha == 0.0:
            x, y, w_roi, h_roi = roi
            undistorted_img = undistorted_img[y:y + h_roi, x:x + w_roi]
        return undistorted_img
