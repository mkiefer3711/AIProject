# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:00:49 2024

@author: Maddison Kiefer
"""

import cv2
import math
import argparse

# Function to highlight faces in the input image
def highlightFace(net, frame, conf_threshold=0.7):
    # Create a copy of the input frame
    frameOpencvDnn = frame.copy()
    # Get the height and width of the frame
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # Create a blob from the frame for the face detection model
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Set the input to the network and forward pass to detect faces
    net.setInput(blob)
    detections = net.forward()
    # List to store the coordinates of detected faces
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # If the confidence of detection is higher than the threshold, consider it as a face
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            # Draw a rectangle around the detected face on the frame
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    # Return the frame with highlighted faces and the coordinates of detected faces
    return frameOpencvDnn, faceBoxes

# Argument parser to accept input image path
parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

# Define the path to the dataset and model files
path = 'C:\\Users\\Maddison\\Downloads\\gad'

faceProto = path + "\\opencv_face_detector.pbtxt"
faceModel = path + "\\opencv_face_detector_uint8.pb"
ageProto = path + "\\age_deploy.prototxt"
ageModel = path + "\\age_net.caffemodel"
genderProto = path + "\\gender_deploy.prototxt"
genderModel = path + "\\gender_net.caffemodel"

# Mean values used for preprocessing the input image for age detection
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Lists to map predicted indices to actual gender and age labels
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load the pre-trained models for face detection, age, and gender classification
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Read the input image
image = cv2.imread(args.image)

# Perform face detection and age/gender classification
resultImg, faceBoxes = highlightFace(faceNet, image)
if not faceBoxes:
    print("No face detected")

# Loop through each detected face
for faceBox in faceBoxes:
    # Extract the face region from the image
    face = image[max(0, faceBox[1] - 20): min(faceBox[3] + 20, image.shape[0] - 1), max(0, faceBox[0] - 20): min(faceBox[2] + 20, image.shape[1] - 1)]
    # Create a blob from the face region for age and gender prediction
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    # Predict gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    # Predict age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    # Draw gender and age on the image
    cv2.putText(resultImg, f'Gender: {gender}, Age: {age[1:-1]} years', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

# Display the result
cv2.imshow("Detecting age and gender", resultImg)
cv2.waitKey(0)
cv2.destroyAllWindows()