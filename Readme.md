# Age and Gender Prediction Model

Here I made use of Deep Neural Network(DNN) for face detection. The function 'faceBox' takes an image frame and a pre-trained face detection model and returns the image frame with bounding boxes drawn around detected faces.

## Process
1. 'faceProto' and 'faceModel' are the file paths to the prototxt and model files for face detection. These files define the architecture and weights of the neural network used for face detection.

2. The 'faceBox' function takes two arguments: 'faceNet', which is a pre-trained face detection network, and 'frame', which is the input image frame where face detection is to be performed.

3. Inside the 'faceBox' function, it first retrieves the dimensions (height and width) of the input frame.

4. It then preprocesses the input frame by converting it into a blob. Blobs are a standard input format for deep learning models. The blob is resized to 227x227 pixels, and normalized, and its color channels are adjusted.

5. The preprocessed blob is set as the input to the 'faceNet' model.

6. The 'faceNet.forward()' method is called to perform ahead pass inference, which results in face detections in the frame.

7. The code initializes an empty list of 'bboxs' to store the coordinates of bounding boxes around detected faces.

8. It loops through the detections from the 'faceNet' model. It extracts the confidence score for each detection and checks if it's above a certain threshold (0.9 in this case) to filter out weak detections.

9. If the confidence is above the threshold, it calculates the coordinates of the bounding box (x1, y1, x2, y2) based on the detection values and scales them according to the frame dimensions.

10. The bounding box coordinates are appended to the 'bboxs' list, and a green bounding box is drawn around the detected face on the input frame using 'cv2.rectangle'.

11. Finally, the function returns the input frame with bounding boxes drawn around detected faces, as well as the list of bounding boxes.

To use this code, you need to load a pre-trained face detection model ('faceNet') using the 'cv2.dnn.readNet' function and then pass video frames or images through the 'faceBox' function to perform face detection and visualize the results with bounding boxes.

## Screenshot

![FR](https://github.com/randomplayer2002/Age-Gender-Prediction/assets/76877728/bc3b406f-fd09-4dc1-95cd-331a2ae217f3)



# Age & Gender Prediction Model 🚀

![FR](https://github.com/randomplayer2002/Age-Gender-Prediction/assets/76877728/bc3b406f-fd09-4dc1-95cd-331a2ae217f3)

A real-time age and gender prediction model using Deep Neural Networks (DNN) with Python and OpenCV. This project detects faces in an image or video stream and predicts the age and gender of each person.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-OpenCV-green.svg" alt="Framework">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

***

## ✨ Features

-   **Real-time Face Detection**: Utilizes a pre-trained Caffe model to detect faces with high accuracy.
-   **Gender Prediction**: Classifies detected faces into 'Male' or 'Female'.
-   **Age Prediction**: Estimates the age of the person from a set of age ranges.
-   **Simple & Efficient**: Built with OpenCV's DNN module for optimized performance.

***

## ⚙️ How It Works

The model operates in a sequential pipeline to process an image or video frame and deliver predictions.

1.  **📥 Input Frame**: The model receives a video frame or a static image.

2.  **👤 Face Detection**:
    -   A pre-trained **SSD (Single Shot Detector) model** with a ResNet-10 base architecture is used for face detection.
    -   The input frame is preprocessed into a blob (resized to 300x300 and normalized).
    -   This blob is passed through the face detection network (`faceNet`) to get potential face locations.

3.  **✅ Confidence Filtering**:
    -   Detections with a confidence score below a certain threshold (e.g., 90%) are filtered out to ensure only high-probability faces are processed.

4.  **🖼️ Bounding Box Extraction**:
    -   For each confident detection, the coordinates of the bounding box around the face are calculated.
    -   A face ROI (Region of Interest) is extracted from the frame using these coordinates.

5.  **🧠 Age & Gender Prediction**:
    -   The extracted face ROI is passed through separate, pre-trained **age and gender prediction networks**.
    -   The networks output the most likely age bracket and gender.

6.  **✏️ Visualization**:
    -   The final frame is annotated with bounding boxes around the detected faces, along with the predicted age and gender labels.

***

## 🛠️ Models Used

This project relies on pre-trained models for each of its core tasks.

| Task               | Model Architecture        | Files Used                                |
| ------------------ | ------------------------- | ----------------------------------------- |
| **Face Detection** | SSD with ResNet-10 base   | `opencv_face_detector.pbtxt`, `opencv_face_detector_uint8.pb` |
| **Age Prediction** | Modified VGG-16           | `age_deploy.prototxt`, `age_net.caffemodel` |
| **Gender Prediction**| Modified VGG-16           | `gender_deploy.prototxt`, `gender_net.caffemodel`|

***

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   Python 3.7 or higher
-   OpenCV library installed
-   A webcam or a video file for real-time prediction

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/Age-Gender-Prediction.git](https://github.com/your-username/Age-Gender-Prediction.git)
    cd Age-Gender-Prediction
    ```

2.  **Install the required libraries:**
    It's recommended to use a virtual environment.
    ```sh
    pip install -r requirements.txt
    ```
    *(If a `requirements.txt` file is not available, install the dependencies manually:)*
    ```sh
    pip install opencv-python numpy
    ```

3.  **Download the pre-trained models** and place them in the appropriate directory within the project.

### Usage

To run the prediction on your webcam feed, execute the following command in your terminal:

```sh
python age_gender_detector.py
