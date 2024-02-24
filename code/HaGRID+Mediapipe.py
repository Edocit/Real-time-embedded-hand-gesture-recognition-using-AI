import cv2
import numpy as np
import time
import keyboard
import mobilenet as mb
import torch
import incV3 as inV
import math
import mediapipe as mp
import HandTrackingModule as htm
from ResNet50 import *
import sys
import os


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

model = None

if(sys.argv[1] == "resnet50"):
    model = ResNet_Custom(50).cuda()
    model = torch.load("resnet50.pth")
if(sys.argv[1] == "mobilenetv3"):
	model = mb.MobileNetV3(18,"large").cuda()
	model.load_state_dict(torch.load("mobnetv3_large_19.pth"))
if(sys.argv[1] == "inceptionv3"):
	model = inV.Inception3(18).cuda()
	model.load_state_dict(torch.load("InceptionV3_12.pth"))

model = model.eval()
time.sleep(3)

classes = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", 
           "peace", "peace inverted", "rock", "stop", "stop inverted", "three", "three2", "two up", "two up inverted"] 


def softmax(vector):
    e = np.exp(vector)
    val = e / e.sum()

    a = [round(x,4) for x in val]

    return a



inference_time = []

best_inference_time = 10e5
worst_inference_time = 0
avg_ingerence_time = 0


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

SCALE = 2
frame_width = 640 * SCALE
frame_height = 480 * SCALE
   
size = (frame_width, frame_height) 
result = cv2.VideoWriter("../recordings/"+sys.argv[3], cv2.VideoWriter_fourcc(*'MJPG'), 20, size) 




ret, frame = cap.read()
initial_rubbish = 0

final_res = -1
filter_window = []
filter_window_size = 5
THD = int(sys.argv[2])/100
handLms = []
TOL = 3

flag = 0

while(True):
    start = time.time()

    ret, frame = cap.read()
    frame_copy = frame.copy()
 
    if(ret):
        #frame = frame.astype(np.float32) / 255.0
        #frame = cv2.resize(frame, (224,224))

        flag = 0
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            flag = 1  
            for handLms in results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                # print(id,lm )
                   h,w,c=frame.shape
                   cx,cy = int(lm.x*w),int(lm.y*h)
                if id == 0:
                    cv2.circle(frame_copy,(cx,cy),15,(0,67,255),cv2.FILLED)
                          
                
        frame_copy = frame.copy().astype(np.uint8)
        frame_copy = cv2.resize(frame_copy, size)
        frame = frame.astype(np.float32) / 255.0
        
        if(sys.argv[1] == "mobilenetv3"):
            frame = cv2.resize(frame, (224,224))
        if(sys.argv[1] == "inceptionv3"):
            frame = cv2.resize(frame, (299,299))
        if(sys.argv[1] == "resnet50"):
            frame = cv2.resize(frame, (224,224))
        
        if(sys.argv[1] == "resnet50"):
        	res = model(torch.from_numpy(frame).unsqueeze(0).permute(0,3,1,2).cuda()) 
        else:
        	res = model(torch.from_numpy(frame).unsqueeze(0).permute(0,3,2,1).cuda()) 
        	
        if(initial_rubbish > 100): inference_time.append((time.time() - start)*1000)

        res = res.detach().cpu().numpy().reshape((18,))
        res = softmax(res)

        if(res[np.argmax(res)] < THD):
            res = -1
        else:
            res = np.argmax(res)

        
        filter_window.append(res)

        if(len(filter_window) == filter_window_size):
            if(filter_window.count(res) >= filter_window_size - TOL):
                final_res = res
            else:
                final_res = -1
            
            filter_window = []

        if(final_res == -1):
            res = "No gesture"
        else:
            res = classes[final_res]

        #frame = cv2.resize(frame, (800,900))
        cv2.putText(frame_copy, res, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if flag == 1:
        	mpDraw.draw_landmarks(frame_copy,handLms,mpHands.HAND_CONNECTIONS)

        result.write(frame_copy)    
        cv2.imshow("DEMO - Deep Learning hand gesture recognition using " + sys.argv[1], frame_copy)
        initial_rubbish += 1
        
        if(cv2.waitKey(5) == ord("q")):
            cv2.destroyAllWindows()
            result.release()
            cap.release()
            break

        #print("TIME", (time.time() - start)*1000)
        #while( (time.time() - start)*1000 < 34): print("ciao")

    else:
        break

result.release() 
cap.release()
cv2.destroyAllWindows()
#for i in range(len(inference_time)):
#    avg_ingerence_time += inference_time[i]
#    if(inference_time[i] < best_inference_time):  best_inference_time  = inference_time[i]
#    if(inference_time[i] > worst_inference_time): worst_inference_time = inference_time[i]

#avg_ingerence_time /= len(inference_time)

# print("Test runtime InceptionV3")
# print("------------------------------------")
# print("Samples:", len(inference_time))
# print("Worst inference:", round(worst_inference_time, 3), "ms")
# print("Avg inference:", round(avg_ingerence_time, 3), "ms")
# print("Best inference:", round(best_inference_time, 3), "ms")
