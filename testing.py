import sys
from TrackingDetection import TrackingDetection
import numpy as np
import cv2

FOCAL_LENGTH = 1460

# Generate a MultiTracker object    
multi_tracker = cv2.MultiTracker_create()
tracker = cv2.TrackerCSRT_create()
 
# Set bounding box drawing parameters
from_center = False # Draw bounding box from upper left
show_cross_hair = False # Don't show the cross hair

videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    print("Failed to open cvcam!!!")
    sys.exit()
# Capture the first video frame
success, frame = videoCapture.read() 
bounding_box_list = []
color_list = []   
 
  # Do we have a video frame? If true, proceed.
if success:
 
    while True:
     
        # Draw a bounding box over all the objects that you want to track_type
        # Press ENTER or SPACE after you've drawn the bounding box
        bounding_box = cv2.selectROI('Multi-Object Tracker', frame, from_center, show_cross_hair) 

        # Add a bounding box
        bounding_box_list.append(bounding_box)
                
        # Add a random color_list
        blue = 255 # randint(127, 255)
        green = 0 # randint(127, 255)
        red = 255 #randint(127, 255)
        color_list.append((blue, green, red))
    
        # Press 'q' (make sure you click on the video frame so that it is the
        # active window) to start object tracking. You can press another key
        # if you want to draw another bounding box.           
        print("\nPress q to begin tracking objects or press " +
            "another key to draw the next bounding box\n")
    
        # Wait for keypress
        k = cv2.waitKey() & 0xFF
    
        # Start tracking objects if 'q' is pressed            
        if k == ord('q'):
            break
    
    cv2.destroyAllWindows()
         
    print("\nTracking objects. Please wait...")

# Add some data 
# mydata[1].update(1)
# mydata[1].update(2)
# mydata[1].update(3)
# mydata[1].update(4)
# print("data 1 average")
# print(mydata[1].get())
# print("data 1")
# mydata[1].print()
# mydata[1].update(5)
# mydata[1].update(6)
# mydata[1].update(7)
# mydata[2].update(8)
# mydata[2].update(9)
# mydata[2].update(10)
# mydata[2].update(11)
# mydata[2].update(12)
# mydata[2].update(13)
# mydata[2].update(14)
# mydata[2].update(15)
# print("data 1")
# mydata[1].print()
# print("data 2")
# mydata[2].print()
# print("data 2 average")
# print(mydata[2].get())
# mydata[1].update(16)
# print("data 1")
# mydata[1].print()
# print("data 1 average")
# print(mydata[1].get())

