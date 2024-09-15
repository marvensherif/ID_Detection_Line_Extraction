import cv2

def rotate_to_make_width_horizontal(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_img = image[y1:y2, x1:x2]
    # Calculate the width and height of the bounding box
    width = x2 - x1
    height = y2 - y1
    # If width is already greater than height, no rotation is needed
    if width >= height:
        print(f"No rotation needed. Width is already greater than height.")
        return cropped_img
    # If height is greater than width, rotate by 90 degrees to make width horizontal
    else:
        # Compute the center of the cropped image
        center = (cropped_img.shape[1] // 2, cropped_img.shape[0] // 2)
        # Apply 90-degree rotation to make the width horizontal
        rotation_matrix = cv2.getRotationMatrix2D(center, 90, scale=1.0)
        rotated_img = cv2.warpAffine(cropped_img, rotation_matrix, (cropped_img.shape[0], cropped_img.shape[1]))
        print(f"Rotated by 90 degrees to make the width parallel to the x-axis.")
        return rotated_img