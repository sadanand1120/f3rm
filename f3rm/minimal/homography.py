import cv2
import numpy as np
import os
import yaml
import math
from copy import deepcopy


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
        if axis == "X":
            mat = [
                [1, 0, 0, 0],
                [0, math.cos(alpha), math.sin(alpha), 0],
                [0, -math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Y":
            mat = [
                [math.cos(alpha), 0, -math.sin(alpha), 0],
                [0, 1, 0, 0],
                [math.sin(alpha), 0, math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Z":
            mat = [
                [math.cos(alpha), math.sin(alpha), 0, 0],
                [-math.sin(alpha), math.cos(alpha), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        else:
            raise ValueError("Invalid axis!")
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
