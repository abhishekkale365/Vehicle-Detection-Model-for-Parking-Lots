import cv2
import numpy as np
import sys


#getting web camera --> No. Since we have a video file
cap = cv2.VideoCapture('video.mp4')
output_file = open('output.txt', 'w')

original_stdout = sys.stdout
sys.stdout = output_file
#min width rectangle
min_width_rect=80 
#min height rectangle
min_height_rect=80

count_line_position = 550
#initialise substructor
algo= cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

detector = []
offset = 6  #allows error between pixel
counter = 0
frame_idx = 0

while True:
    ret,frame1= cap.read()          # Ret will return true or false if it has found the frame or not.
    if not ret:
        break
    frame_idx += 1
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilated = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #debug = frame1.copy()
    #cv2.drawContours(debug, counterShape, -1, (0,255,0), 2)
    #cv2.imshow("Contours", debug)

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rect) and (h>= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w, y+h),(0,225,0),2)
        cv2.putText(frame1, "Vehicle : "+ str(counter), (x,y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,244,0), 2)

        center = center_handle(x,y,w,h)
        detector.append(center)
        print(f"[f{frame_idx}] + added center: ({center[0]},{center[1]})  | detector size now: {len(detector)}")
        #print(detector)
        cv2.circle(frame1,center,4,(0,0,255),-1)
        #cv2.imshow("Centers Visualisation", frame1)
   
        new_detector = []
        band_lo = count_line_position - offset
        band_hi = count_line_position + offset
        print(f"[f{frame_idx}] counting band: y in [{band_lo}, {band_hi}]")

        for (cx, cy) in detector:
            if band_lo <= cy <= band_hi:
                counter += 1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0,127,255), 3)
                cv2.rectangle(frame1, (25, count_line_position - offset), (1200, count_line_position + offset), (0, 255, 255), 1)  # thin yellow band
                print(f"[f{frame_idx}]  (tick) counted center ({cx},{cy}) -> counter={counter}")
            else:
                new_detector.append((cx, cy))
                print(f"[f{frame_idx}] --> keep center   ({cx},{cy}) for next frame")

        detector = new_detector
        print(f"[f{frame_idx}] detector kept: {len(detector)} items\n")

        #print("Vehilce Counter : "+ str(counter))

    
    cv2.putText(frame1, "VEHICLE COUNTER : "+ str(counter), (450,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)

   
    #cv2.imshow('Detector', dilated)
    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(10) == 13:
        break

cv2.destroyAllWindows()
cap.release()
sys.stdout = original_stdout
output_file.close()
