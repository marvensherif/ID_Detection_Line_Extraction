import cv2
import numpy as np


def align_image_to_reference(reference_img, target_img, min_matches=10):
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray_target, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_ref, descriptors_target)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < min_matches:
        return None, matches, keypoints_ref, keypoints_target
    ref_points = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target_points = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography_matrix, mask = cv2.findHomography(target_points, ref_points, cv2.RANSAC, 5.0)
    h, w = reference_img.shape[:2]
    aligned_img = cv2.warpPerspective(target_img, homography_matrix, (w, h))

    return aligned_img, matches, keypoints_ref, keypoints_target


