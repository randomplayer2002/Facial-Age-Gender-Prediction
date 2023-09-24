import cv2

faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'

def faceBox(faceNet,frame):
    # Get frame dimensions
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0 , (227,227), [104,117,123], swapRB = False)
    
    # Set blob as input to faceNet
    faceNet.setInput(blob)

    # Get detection from faceNet
    detection = faceNet.forward()

    # Initialize list to store bounding boxes
    bboxs = []

    # Loop through detections
    for i in range(detection.shape[2]):

        #Get confidence of detection
        confidence = detection[0,0,i,2]

        # Check if confidence is above threshold
        if confidence > 0.9:

            # Get bounding box coordinates
            x1 = int(detection[0,0,i,3]*frame_width)
            y1 = int(detection[0,0,i,4]*frame_height)
            x2 = int(detection[0,0,i,5]*frame_width)
            y2 = int(detection[0,0,i,6]*frame_height)

            # Append bounding box to list
            bboxs.append([x1,y1,x2,y2])

            # Draw bounding box on frame
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)

    # Return frame with bounding boxes and list of bounding boxes        
    return frame, bboxs
