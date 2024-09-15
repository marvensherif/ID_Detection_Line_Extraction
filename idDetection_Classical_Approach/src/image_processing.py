import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import math

def load_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (800, 800))
    return image

def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred
    
def edge_detection(image):
    edges = cv2.Canny(image,50,150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour
    
def detect_corners_from_contour(contours):
    epsilon = 0.02 * cv2.arcLength(contours, True)
    approx_corners = cv2.approxPolyDP(contours, epsilon, True)
    return approx_corners

def sort_coordinates_x_y(coordinates):
    points = np.array(coordinates)
    points = np.squeeze(points,1)
    sorted_points = sorted(points, key=lambda point: (point[0], point[1]))
    return np.array(sorted_points)

def angle_of_line(x1,x2,y1,y2):
    return math.degrees(math.atan2(-(y2-y1),x2-x1))
    
def find_top_corners(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    top_leftness = points[:, 0] + points[:, 1]  
    top_rightness = points[:, 0] - points[:, 1]  
    top_left_index = np.argmin(top_leftness)
    top_left = points[top_left_index]
    sorted_indices = np.argsort(top_rightness)
    top_half_indices = sorted_indices[-2:] 
    top_right = points[top_half_indices[np.argmin(points[top_half_indices, 1])]] 
    return top_left, top_right
    
def rotate_image(image,points):
    top_left,top_right = find_top_corners(points)
    x1,y1 = top_left
    x2,y2 = top_right
    rotate_angle = angle_of_line(x1,x2,y1,y2)
    rotation_transform = A.Compose([
        A.Rotate(limit=(-rotate_angle, -rotate_angle), p=1.0) 
    ],keypoint_params = A.KeypointParams(format='xy'),)
    t = rotation_transform(image=image,keypoints=points)
    rotated_img = t['image']
    keypoints = t['keypoints']
    return rotated_img,keypoints

def crop_region_id(image,keypoints):
    x, y, w, h = cv2.boundingRect(np.array(keypoints)) 
    id_region = image[y:y+h, x:x+w]
    return id_region

def preprocess_and_segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    region_id_2 = img.copy()
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        line_image = region_id_2[y:y+h, x:x+w]
        cv2.rectangle(region_id_2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return region_id_2