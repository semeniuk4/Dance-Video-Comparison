import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def normalize_angles(angles_dict):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_angles_dict = {}
    for key in angles_dict:
        angles = np.array(angles_dict[key]).reshape(-1, 1)
        normalized_angles_dict[key] = scaler.fit_transform(angles).flatten()
    return normalized_angles_dict

    
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    angles_dict = {
        'right_shoulder_elbow_wrist': [],
        'left_shoulder_elbow_wrist': [],
        'right_elbow_shoulder_left_shoulder': [],
        'left_elbow_shoulder_right_shoulder': [],
        'left_shoulder_hip_knee': [],
        'right_shoulder_hip_knee': [],
        'right_hip_knee_ankle': [],
        'left_hip_knee_ankle': []
    }
    
    
    with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
        if not cap.isOpened():
            print("Could not open video.")
            return angles_dict
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                angles_dict['right_shoulder_elbow_wrist'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]))  
                
                angles_dict['left_shoulder_elbow_wrist'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]))  
                                
                angles_dict['right_elbow_shoulder_left_shoulder'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]))  
                
                angles_dict['left_elbow_shoulder_right_shoulder'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]))  
                                
                angles_dict['left_shoulder_hip_knee'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]))  
                
                angles_dict['right_shoulder_hip_knee'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]))  
                  
                angles_dict['right_hip_knee_ankle'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],  
                    [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]))  
                
                angles_dict['left_hip_knee_ankle'].append(calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],  
                    [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]))  
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Video Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        angles_dict = normalize_angles(angles_dict)
        
    return angles_dict

def compare_videos(path1, path2):
    angles_video1 = process_video(path1)
    angles_video2 = process_video(path2)
    
    scores = {}
    print("\n")
    print('Calculating the result...')
    for key in angles_video1:
        series1 = np.array(angles_video1[key]).reshape(-1, 1)
        series2 = np.array(angles_video2[key]).reshape(-1, 1)
        distance = dtw.distance(series1, series2)
        scores[key] = 1 / (1 + distance)
    return scores

video_path1 = '/Users/ivansemeniuk/Downloads/studio-3.mov'
video_path2 = '/Users/ivansemeniuk/Downloads/calm-down-2.mp4'

scores = compare_videos(video_path1, video_path2)

print(f"Similarity of right hand movements: {scores['right_shoulder_elbow_wrist']*100}%")
print(f"Similarity of left hand movements: {scores['left_shoulder_elbow_wrist']*100}%")
print(f"Similarity of movements of the right arm in the shoulder: {scores['right_elbow_shoulder_left_shoulder']*100}%")
print(f"Similarity of movements of the left arm in the shoulder: {scores['left_elbow_shoulder_right_shoulder']*100}%")
print(f"Similarity of movements of the right side of the body: {scores['right_shoulder_hip_knee']*100}%")
print(f"Similarity of movements of the left side of the body: {scores['left_shoulder_hip_knee']*100}%")
print(f"Similarity of movements of the right leg: {scores['right_hip_knee_ankle']*100}%")
print(f"Similarity of movements of the left leg: {scores['left_hip_knee_ankle']*100}%")
print(f"General score: {sum(scores.values())/8*100}%")