## id_detecetion_and_line_extraction_deeplearning_Approach

The deep learning approach utilizes two fine-tuned YOLO models for accurate and efficient 
ID card detection and text line segmentation
YOLO Model 1: ID Card Detection
YOLO Model 2: Text Line Segmentation and Labeling.

## Structure

1. `src/`: Contains the main script and image processing functions.
2. `test_images/`: Input folder with test images.
3. `output_images/`: Output folder for processed images (generated after running the project).
4. `weights/model1`: Contains the YOLO model weights for id detection.
5. `weights/model2`: Contains the YOLO model weights fro line extraction.
6. `requirements.txt`: List of dependencies.
7. `Dockerfile`: to build docker image.

## Setup Instructions
1. cd idDetection_DeepLearning_Approach
2. Install the required dependencies:
    pip install -r requirements.txt
3. Place your test images in the `test_images/` folder.
4. Run the main script:
    python src/main.py

## Running using docker
1. cd idDetection_DeepLearning_Approach
2. Place your test images in the `test_images/` folder.
3. docker build -t your_image_name .
4. docker run -it your_image_name
5. python src/main.py
6. docker cp <container_id>:/app/output_images ./output_images
- **get container_id from docker ps command**

The Processed images will be saved in the `output_images/` folder.

