import os
from image_processing import load_image, preprocessing, edge_detection, detect_corners_from_contour, rotate_image, crop_region_id, preprocess_and_segment,sort_coordinates_x_y
import cv2


def pipeline(image_path, output_path):
    img = load_image(image_path)
    processed_img = preprocessing(img)
    largest_contour = edge_detection(processed_img)
    points = detect_corners_from_contour(largest_contour)
    sorted_points = sort_coordinates_x_y(points)
    rotated_img, keypoints = rotate_image(img, sorted_points)
    region_id = crop_region_id(rotated_img, keypoints)
    segmented_img = preprocess_and_segment(region_id)
    cv2.imwrite(output_path, segmented_img)
            
            
def main():
    input_folder = "./test"
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")
            pipeline(image_path, output_path)

if __name__ == "__main__":
    main()

