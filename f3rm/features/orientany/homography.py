import math
from copy import deepcopy
import cv2
import numpy as np


class Homography:
    def __init__(self):
        pass

    @staticmethod
    def projectCCStoPCS(ccs_coords, K, width, height, d=None, mode="skip"):
        """
        Projects set of points in CCS to PCS.
        ccs_coords: (N x 3) array of points in CCS
        Returns: (N x 2) array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
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
    def general_project_A_to_B(inp, AtoBmat):
        """
        Project inp from A frame to B
        inp: (N x 3) array of points in A frame
        AtoBmat: (4 x 4) transformation matrix from A to B
        Returns: (N x 3) array of points in B frame
        """
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
    def get_std_trans(cx=0, cy=0, cz=0):
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
        return np.array(mat)

    @staticmethod
    def get_std_rot(axis, alpha):
        """
        axis is either "X", "Y", or "Z" axis of F and alpha is positive acc to right hand thumb rule dirn
        Multiplication goes like M_coords = T * F_coords
        """
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

        return np.array(mat)
