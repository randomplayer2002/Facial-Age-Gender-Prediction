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
