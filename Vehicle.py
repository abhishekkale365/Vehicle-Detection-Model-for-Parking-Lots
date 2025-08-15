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

up_count = 0     # vehicles moving up (y decreases across the line)
down_count = 0   # vehicles moving down (y increases across the line)   In open CV y increases downwards

count_line_position = 550
#initialise substructor
algo= cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

prev_centers = []           # centers from the previous frame
recent = []                 # list of (x, y, ttl) to avoid double-counts if a car lingers in the band
RECENT_TTL = 8              # frames to suppress re-counts for the same spot
RADIUS = 120                 # px radius to consider two points "the same" for debounce / matching
detector = []
offset = 6                 #allows error between pixel
counter = 0
frame_idx = 0

while True:
    ret,frame1= cap.read()          # Ret will return true or false if it has found the frame or not.
    if not ret:
        break
    # decay recent counted spots (debounce)
    recent = [(x, y, ttl-1) for (x, y, ttl) in recent if ttl-1 > 0]

    # remember previous frame's centers and start a fresh list for this frame
    last_prev = prev_centers[:]
    current_centers = []
    frame_idx += 1

    #Greying and Blurring out the centres
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
        current_centers.append(center)
        print(f"[f{frame_idx}] + added center: ({center[0]},{center[1]})  | current size now: {len(current_centers)}")
        cv2.circle(frame1, center, 4, (0,0,255), -1)

   
        # visualize the band (optional, helps tuning)
        band_lo = count_line_position - offset
        band_hi = count_line_position + offset
        cv2.rectangle(frame1, (25, band_lo), (1200, band_hi), (0, 255, 255), 1)

        def nearest_prev(cx, cy, prev_points, max_r=RADIUS):
            """Return (px, py) of the nearest previous point within max_r, else None."""
            best = None
            best_d2 = max_r * max_r
            for (px, py) in prev_points:
                d2 = (cx - px)**2 + (cy - py)**2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (px, py)
            return best

        for (cx, cy) in current_centers:
            # only consider counts inside the band
            if band_lo <= cy <= band_hi:
                # debounce: skip if we recently counted the same spot
                is_repeat = False
                for i, (rx, ry, ttl) in enumerate(recent):
                    if (cx - rx)**2 + (cy - ry)**2 <= RADIUS * RADIUS:
                        # refresh TTL, skip counting
                        recent[i] = (rx, ry, RECENT_TTL)
                        is_repeat = True
                        break
                if is_repeat:
                    continue
                
                # decide direction using nearest previous center
                prev = nearest_prev(cx, cy, last_prev, max_r=RADIUS)
                if prev is not None:
                    px, py = prev
                    dy = cy - py  # +ve => moving DOWN (y increases), -ve => moving UP
                    print(f"[f{frame_idx}] dir-check match: prev=({px},{py}) now=({cx},{cy}) dy={dy}")
                    cv2.rectangle(frame1, (25, band_lo), (1200, band_hi), (0,255,255), 1)

                    # When prev exists:
                    cv2.arrowedLine(frame1, (px, py), (cx, cy), (255,0,0), 2, tipLength=0.3)
                    if dy > 0:
                        down_count += 1
                        dir_txt = "DOWN"
                        flash_color = (0, 180, 255)
                    elif dy < 0:
                        up_count += 1
                        dir_txt = "UP"
                        flash_color = (0, 255, 180)
                    else:
                        dir_txt = "FLAT"
                        flash_color = (128, 128, 128)
                else:
                    # no good previous match → count as total only (direction unknown)
                    dir_txt = "UNK"
                    flash_color = (200, 200, 200)
                    print(f"[f{frame_idx}] dir-check NO MATCH for ({cx},{cy}) within RADIUS={RADIUS}")

                # increment total and mark this spot as recently counted
                counter += 1
                recent.append((cx, cy, RECENT_TTL))

                # visual feedback
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), flash_color, 3)
                cv2.putText(frame1, dir_txt, (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, flash_color, 2)


            cv2.putText(frame1, f"TOTAL: {counter}", (40, 60),cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)

            cv2.putText(frame1, f"DOWN:  {down_count}", (40, 110),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

            cv2.putText(frame1, f"UP:    {up_count}", (40, 150),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 180), 3)
   


            prev_centers = current_centers
            #cv2.imshow('Detector', dilated)
            cv2.imshow('Video Original',frame1)

    if cv2.waitKey(120) == 13:
        break

cv2.destroyAllWindows()
cap.release()
sys.stdout = original_stdout
output_file.close()
