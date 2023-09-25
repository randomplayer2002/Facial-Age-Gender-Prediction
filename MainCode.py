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

ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'

genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)
age_list = ['(0-2)','(4-6)', '(8-12)', '(15-20)','(21-32)','(38-43)','(48-53)','(60-100)']
gender_list = ['Male', 'Female']
model_mean_value = (78.4263377603,87.7689143744, 114.895847746)
video  = cv2.VideoCapture(0)
padding = 20
    
while True:
    ret,frame= video.read()
    frame,bboxs = faceBox(faceNet,frame)
    for bbox in bboxs:
        face= frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face,1.0 , (227,227),model_mean_value,swapRB = False)
        
        
        genderNet.setInput(blob)
        gender_pred = genderNet.forward()
        gender = gender_list[gender_pred[0].argmax()]
        
        
        ageNet.setInput(blob)
        age_pred = ageNet.forward()
        age = age_list[age_pred[0].argmax()]
        
        label = "{},{}".format(gender,age)
        cv2.rectangle(frame, (bbox[0], bbox[1]-10), (bbox[2], bbox[1]),(0,255,0),-1)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2, cv2.LINE_AA)
        
        
    cv2.imshow('ProjectGurukul Age-Gender', frame)
    k = cv2.waitKey(1)
    if k==ord('x'):
        break

video.release()
cv2.destroyAllWindows()
