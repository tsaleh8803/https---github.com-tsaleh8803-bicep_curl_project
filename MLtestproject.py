import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import csv
from landmarks import landmarks
from sklearn.model_selection import train_test_split

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
EXPORT_PATH = "bicep_curl_dataset.csv"
MODEL_PATH = "bicep.pkl"

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

counter = 0
stage = None
prob = np.array([0,0]) 

#Retrieve model from picke file
with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)
#initiate holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #recoloring image since frame is in BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #make detections
        results = holistic.process(image)
        image_height, image_width, _ = image.shape
        #convert image back to BGR to process it
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)        

        try:    

            landmark_list = results.pose_landmarks.landmark
            shoulder = [landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmark_list[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmark_list[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmark_list[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmark_list[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = int(calculate_angle(shoulder, elbow, wrist))
            
            row = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            row = np.insert(row,0,angle)
            X = pd.DataFrame([row],columns = landmarks[1:])
            
            predicted_stage = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            # print(predicted_stage)
            # print(prob)
            if predicted_stage == "down" and prob.max() >= 0.7:
                stage = "down"
            elif stage == "down" and predicted_stage == "up" and prob.max() >= 0.7:
                stage = "up"
                counter += 1
        except:
            pass

        #Body Drawings
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

        cv2.rectangle(image,(0,0),(800,140),(245,117,56),-1)
        
        #Rep Display
        cv2.putText(image, 'REPS', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image, str(counter), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        #Stage Display
        cv2.putText(image, 'STAGE', (250,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image, stage, (250,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        #Probability Display
        cv2.putText(image, 'PROB', (450,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image, str(prob.max()), (450,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam Footage',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

