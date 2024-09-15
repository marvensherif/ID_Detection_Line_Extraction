## id_detecetion_and_line_extraction_Classsical_Approach

The Classical approach detect the ID card and segment the text lines using traditional computer vision techniques
such as edge detection and corner detection.

## Structure

1. `src/`: Contains the source code for the project.
2. `test/`: Contains the test images.
3. `output/`: Folder where the processed images will be saved (generated after running the project).
4. `requirements.txt`: List of dependencies.
5. `Dockerfile`: to build docker image.

## Setup Instructions
1. cd idDetection_Classical_Approach
2. Install the required dependencies:
     pip install -r requirements.txt
3. Place your test images in the `test/` folder.
4. Run the main script:
      python src/main.py

## Running using docker
1. cd idDetection_Classical_Approach
2. Place your test images in the `test/` folder.
3. docker build -t your_image_name .
4. docker run -it your_image_name
5. python src/main.py
6. docker cp <container_id>:/app/output ./output
- **get container_id from docker ps command**

The processed images will be saved in the output/ folder.
