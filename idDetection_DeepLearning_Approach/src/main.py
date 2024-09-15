import os
import cv2
from ultralytics import YOLO
from image_processing import rotate_to_make_width_horizontal

def pipeline(folder_path, model1, model2, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    # Run batched inference using model1 to detect the ID card
    results1 = model1(image_paths)
    for idx, result1 in enumerate(results1):
        img_path = image_paths[idx]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        boxes = result1.boxes
        if len(boxes) > 0:
            rotated_img = rotate_to_make_width_horizontal(img, boxes[0])
            # Now run model2 on the rotated image to get better object detection
            results2 = model2([rotated_img])
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