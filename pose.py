import cv2
# from os import kill
import config1
import mediapipe as mp
import numpy as np
import math as math
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose


# kill_cam = False

count = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
count_6 = 0
count_7 = 0 

stage = None
stage_2 = None
stage_3 = None
stage_4 = None
stage_5 = None
stage_6 = None
stage_7 = None

d = [0, 0, 0, 0, 0, 0, 0]

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def findAngle(x1, y1, x2, y2):
    theta = math.acos( (y2 -y1)*(-y1) / (math.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/math.pi)*theta
    return degree

def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def finished():

    config1.kill_cam = True
    # kill_cam = True
    return "kill_webcam"


outputFrame = None

# cap = cv2.VideoCapture(0)
## Setup mediapipe instance
def video():
    d = [0, 0, 0, 0, 0, 0, 0]
    count = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0   

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose 
    
    # cap = cv2.VideoCapture(2) 
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
        # while True:
            if config1.kill_cam == True:
                
                for i in range (10):
                    print("config1.kill no IF")
                              
                break
                # return d

            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # image = cv2.resize(image, (1280, 720))
            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # h, w = image.shape[:2]
                       
            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_L = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_L = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                eye_LI = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]
                ear_L = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_R = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                eye_RI = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
                ear_R = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                ang_elb_l  = round(calculate_angle(shoulder_L, elbow_L, wrist_L),2)

                ang_elb_R  = round(calculate_angle(shoulder_R, elbow_R, wrist_R),2)

                ang_sh_l   = round(calculate_angle(elbow_L,shoulder_L,hip_L),2)

                ang_sh_R   = round(calculate_angle(elbow_R,shoulder_R,hip_R),2)

                ang_hip_l  = round(calculate_angle(shoulder_L,hip_L,knee_L),2)

                ang_hip_R  = round(calculate_angle(shoulder_R,hip_R,knee_R),2)

                ang_knee_l = round(calculate_angle(hip_L,knee_L,ankle_L),2)

                ang_knee_R = round(calculate_angle(hip_R,knee_R,ankle_R),2)
                
                # middle_sh = findDistance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                # middle_M = findDistance(landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y)
                neck_angle = round(findAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y),2)
                torso_angle_L = round(findAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),2)
                torso_angle_R = round(findAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),2)
                
                
                cv2.putText(image, str(neck_angle),   
                            tuple(np.multiply(ear_R, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(ang_elb_l),   
                            tuple(np.multiply(elbow_L, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
            
                cv2.putText(image, str(ang_elb_R), 
                            tuple(np.multiply(elbow_R, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                
                cv2.putText(image, str(ang_sh_l), 
                            tuple(np.multiply(shoulder_L, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
            
                cv2.putText(image, str(ang_sh_R), 
                            tuple(np.multiply(shoulder_R, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(torso_angle_L), 
                            tuple(np.multiply(hip_L, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
            
                cv2.putText(image, str(torso_angle_R), 
                            tuple(np.multiply(hip_R, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )     

                cv2.putText(image, str(ang_knee_l), 
                            tuple(np.multiply(knee_L, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
            
                cv2.putText(image, str(ang_knee_R), 
                            tuple(np.multiply(knee_R, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )      

                # #  LEFT SHOULDER 
                
                # # if ang_sh_l > 100:
                # #     stage = "down"
                # # if ang_sh_l <  90 and stage =='down':
                # #     stage="up"
                # #     count +=1
                    
                # if ang_sh_l < 95:
                #     stage = "down"
                # if ang_sh_l >  100 and stage =='down':
                #     stage="up"
                #     count +=1    

                # #  RIGHT SHOULDER
                
                # # if ang_sh_R > 100:
                # #     stage_2 = "down"
                # # if ang_sh_R < 90 and stage_2 =='down':
                # #     stage_2="up"
                # #     count_2 +=1
                                     
                # if ang_sh_R < 95:
                #     stage = "down"
                # if ang_sh_R >  100 and stage =='down':
                #     stage="up"
                #     count +=1 
                

                # #  LEFT ELBOW
                # # if ang_elb_l > 140:
                # #     stage_3 = "down"
                # # if ang_elb_l  < 90 and stage_3 =='down':
                # #     stage_3="up"
                # #     count_3 +=1
                
                # if ang_elb_l < 135:
                #     stage_3 = "down"
                # if ang_elb_l  > 160 and stage_3 =='down':
                #     stage_3="up"
                #     count_3 +=1

                # #  RIGHT ELBOW
                
                # # if ang_elb_R > 140:
                # #     stage_4 = "down"
                # # if ang_elb_R < 90 and stage_4 =='down':
                # #     stage_4="up"
                # #     count_4 +=1
                    
                # if ang_elb_R < 135:
                #     stage_3 = "down"
                # if ang_elb_R  > 160 and stage_3 =='down':
                #     stage_3="up"
                #     count_3 +=1
                
                # # #  counter logic
                # # if ang_hip_l > 120:
                # #     stage_5 = "down"
                # # if ang_hip_l < 60 and stage_5 =='down':
                # #     stage_5 ="up"
                # #     count_5 +=1
                    
                # # #  counter logic
                # # if ang_hip_R > 120:
                # #     stage_6 = "down"
                # # if ang_hip_R < 60 and stage_6 =='down':
                # #     stage_6="up"
                # #     count_6 +=1

                # #  LEFT HIP 
                
                # if torso_angle_L < 10:
                #     stage_5 = "down"
                # if torso_angle_L >15 and stage_5 =='down':
                #     stage_5 ="up"
                #     count_5 +=1
                    
                # #  RIGHT HIP 
                
                # if torso_angle_R < 10:
                #     stage_6 = "down"
                # if torso_angle_R > 15 and stage_6 =='down':
                #     stage_6="up"
                #     count_6 +=1    
                    
                # # NECK --> RIGHT
                
                # # if neck_angle < 30:
                # #     stage_7 = "down"
                # # if neck_angle < 20 and stage_7 =='down':
                # #     stage_7="up"
                # #     count_7 +=1

                # if neck_angle < 20:
                #     stage_7 = "down"
                # if neck_angle > 30 and stage_7 =='down':
                #     stage_7="up"
                #     count_7 +=1

# ============================================================================
 
 
                 #  RIGHT ELBOW
                                                   
                if ang_elb_R < 135:
                    stage_4 = "down"
                if ang_elb_R  > 140 and stage_4 =='down':
                    stage_4="up"
                    count_4 +=1 
                      
                #  LEFT ELBOW
                                
                if ang_elb_l < 135:
                    stage_3 = "down"
                if ang_elb_l  > 140 and stage_3 =='down':
                    stage_3="up"
                    count_3 +=1
                    
                
                #  RIGHT SHOULDER
                
                if ang_sh_R > 100:
                    stage_2 = "down"
                if ang_sh_R < 90 and stage_2 =='down':
                    stage_2="up"
                    count_2 +=1   
                
                
                 #  LEFT SHOULDER 
                
                if ang_sh_l > 100:
                    stage = "down"
                if ang_sh_l <  90 and stage =='down':
                    stage="up"
                    count +=1
                    
                               
                      
                # NECK --> LEFT
                
                if neck_angle < 30:
                    stage_7 = "down"
                if neck_angle > 40 and stage_7 =='down':
                    stage_7="up"
                    count_7 +=1   
                    
                 #  RIGHT HIP 
                
                if torso_angle_R < 10:
                    stage_6 = "down"
                if torso_angle_R > 15 and stage_6 =='down':
                    stage_6="up"
                    count_6 +=1      
                                               
                #  LEFT HIP 
                
                if torso_angle_L < 10:
                    stage_5 = "down"
                if torso_angle_L >15 and stage_5 =='down':
                    stage_5 ="up"
                    count_5 +=1
                        
               
                cv2.putText(image, str(count), 
                        (45,45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (30,144,255), 2, cv2.LINE_AA)
            
                cv2.putText(image, str(count_2), 
                        (45,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (30,144,255), 2, cv2.LINE_AA)

                cv2.putText(image, str(count_3), 
                        (45,155), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (30,144,255), 2, cv2.LINE_AA)

                cv2.putText(image, str(count_4), 
                        (45,210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (30,144,255), 2, cv2.LINE_AA)
            
                cv2.putText(image, str(count_5), 
                        (45,265), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (30,144,255), 2, cv2.LINE_AA)

                cv2.putText(image, str(count_6), 
                        (45,320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (30,144,255), 2, cv2.LINE_AA)
                
                cv2.putText(image, str(int(count_7)), 
                        (45,370), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (30,144,255), 2, cv2.LINE_AA)
                
                
                # angle_text_string = 'Neck : ' + str(int(neck_angle)) + '  Torso esquerdo: ' + str(int(torso_angle_L))+ '  Torso direito : ' + str(int(torso_angle_R))
                # print(angle_text_string)

            
                # # Join landmarks.
                
                # cv2.line(image, middle_M, middle_sh,(0,255,0),5)
                # cv2.line(image, (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y), (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y), green, 4)
                # cv2.line(image, (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y), (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y - 100), green, 4)
                # cv2.line(image, (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y), (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y), green, 4)
                # cv2.line(image, (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y), (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - 100), green, 4)
            
            except:
                pass
        
        # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=4) 
                                    )               


            d = [count, count_2, count_3, count_4, count_5, count_6, count_7]
            print(d)
            
            global outputFrame
            outputFrame = image.copy()
            # cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
                
    return d       


def generateImages():
	global outputFrame
	while True:
		if outputFrame is None:
			continue
		(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
		if not flag:
			continue
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')