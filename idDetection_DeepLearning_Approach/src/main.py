import os
import cv2
from ultralytics import YOLO
from image_processing import align_image_to_reference

def pipeline(folder_path, model1, model2, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
     # Load the reference image
    reference_image_1 = cv2.imread("ref.jpg")
    reference_image_2 = cv2.imread("ref2.jpg")
    # Run batched inference using model1 to detect the ID card
    results1 = model1(image_paths)
    for idx, result1 in enumerate(results1):
        img_path = image_paths[idx]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        boxes = result1.boxes
        if len(boxes) > 0:
            # Assuming the first box is the ID card
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            # Try aligning the cropped image with both reference_image_1 and reference_image_2
            aligned_img_1, matches_1, keypoints_ref_1, keypoints_target_1 = align_image_to_reference(reference_image_1, cropped_img)
            aligned_img_2, matches_2, keypoints_ref_2, keypoints_target_2 = align_image_to_reference(reference_image_2, cropped_img)
            # Choose the best alignment based on the number of matches
            if len(matches_1) >= len(matches_2):
                print(f"Using reference_image_1 with {len(matches_1)} matches.")
                aligned_img = aligned_img_1
            else:
                print(f"Using reference_image_2 with {len(matches_2)} matches.")
                aligned_img = aligned_img_2
            # If no alignment (i.e., not enough matches), skip this image
            if aligned_img is None:
                print(f"Alignment with both references failed for {img_name}. Skipping.")
                continue  # Skip this image if both reference alignments fail
            results2 = model2([aligned_img])
            # Draw the bounding boxes from model2 and save the final output
            for result2 in results2:
                annotated_img = result2.plot()  # Draw bounding boxes and annotations
                final_img_path = os.path.join(output_folder, f"final_{img_name}")
                cv2.imwrite(final_img_path, annotated_img)

if __name__ == "__main__":
    # Load YOLO models
    model1 = YOLO("weights/model1/best.pt")  # Path to model1 weights for ID card detection
    model2 = YOLO("weights/model2/best.pt")  # Path to model2 weights for line segmentation

    pipeline("test_images", model1, model2, "output_images")