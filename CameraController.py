import numpy as np
import argparse
import imutils
import cv2
import os
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

def capture_image():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    #initialize a total amount of pictures taken for data
    img_counter = 0
    xframe = 600
    yframe = 400
    xmid = (xframe/2)
    ymid = (yframe/2)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()



    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame = cv2.resize(frame,(xframe,yframe))
        cv2.putText(frame,".",(int(xmid),int(ymid)),cv2.FONT_HERSHEY_COMPLEX,(3),(0,0,255))

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)


            while(os.path.isfile("opencv_frame_{}.png".format(img_counter))):
                img_counter += 1
           
            img_name = "opencv_frame_{}.png".format(img_counter)

            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))

            return img_name


        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = capture_image()
    print(img)