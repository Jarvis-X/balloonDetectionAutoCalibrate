#!/usr/bin/env python3


# import the necessary packages
import time
import cv2
from cv2 import threshold
from matplotlib.collections import TriMesh
import numpy as np
import sys
import pickle
import os

from sympy import Q
from TrackingDetection import TrackingDetection


# Constant for focal length, in pixels (must be changed per camera)
SQUAREWIDTH = 1.60
TRIANGLEWIDTH = 1.96 # 2/sq(3)*170
BALLOONWIDTH = 0.33

FRAMEX = 600
FRAMEY = 600

rospy.init_node("balloon_detection", anonymous=True)
rate = rospy.Rate(5)
balloonpub = rospy.Publisher('balloon', Float64MultiArray, queue_size=1)
goalpub = rospy.Publisher('target', Float64MultiArray, queue_size=1)
framepub = rospy.Publisher('frameinfo', Float64MultiArray, queue_size=1)


def parse_args():
    """Function to parse the system arguments"""
    # mode[0] defines frame obtaining methods - 0: picamera; 1: opencv camera
    # mode[1] defines operation mode - 0: detection mode; 1: calibration mode
    mode = [1, 0]
    args = sys.argv
    if len(args) == 5:
        if args[3] == "__name:=cvMain":
            args = args[0:3]
    if len(args) < 2:
        print("ERROR: Insufficient arguments",
              "Please use the following form:",
              "python main.py <picam/cvcam> <1 if calibration mode else leave blank or input anything to nullify>")
        sys.exit()
    elif len(args) > 3:
        print("error:", args)
        print("ERROR: Too many arguments",
              "Please use the following form:",
              "python main.py <picam/cvcam> <1 if calibration mode else leave blank or input anything to nullify>")
        sys.exit()
    else:
        if args[1] == "picam":
            print("picamera in progress, exiting!")
            mode[0] = 0
            if len(args) == 3:
                if int(args[2]) == 0:
                    mode[1] = 0
                    print("Balloon detection blob detection mode && Target detection webcam.")
                elif int(args[2]) == 1:
                    mode[1] = 1
                    print("Color calibration mode, please place the balloon in the center of the frame")
                elif int(args[2]) == 4:
                    print("Color calibration mode, please place the goal in the center of the frame.")
                    mode[1] = 4
                else:
                    print("Invalid mode for now, exiting!")
                    sys.exit()
        else:
            assert args[1] == "cvcam"
            mode[0] = 1
            if len(args) == 3:
                if int(args[2]) == 0:
                    mode[1] = 0
                    print("Balloon detection blob detection mode && Target detection color mode.")
                elif int(args[2]) == 1:
                    mode[1] = 1
                    print("Color calibration mode, please place the balloon in the center of the frame")
                elif int(args[2]) == 2:
                    print("Balloon detection color filter mode.")
                    mode[1] = 2
                elif int(args[2]) == 3:
                    print("Balloon detection blob detection mode.")
                    mode[1] = 3
                elif int(args[2]) == 4:
                    print("Color calibration mode, please place the goal in the center of the frame.")
                    mode[1] = 4
                elif int(args[2]) == 5:
                    print("Target detection canny mode.")
                    mode[1] = 5
                elif int(args[2]) == 6:
                    print("Target detection color mode.")
                    mode[1] = 6
                else:
                    print("Invalid mode, exiting!")
                    sys.exit()
            else:
                print("Invalid mode for now, exiting!")
                sys.exit()
    return mode


def n_channel_min_mean_max(n_channel_new_data, frame_count, n_channel_data_array, n=3):
    for i in range(n):
        data_min_mean_max(n_channel_new_data[:, :, i], frame_count, n_channel_data_array[i])


def data_min_mean_max(new_data, frame_count, data_array):
    mean_val = np.mean(new_data)
    # print(mean_val, data_array[0])
    if data_array[0] > mean_val:
        data_array[0] = mean_val
    if data_array[2] < mean_val:
        data_array[2] = mean_val
    data_array[1] = data_array[1] + (mean_val - data_array[1]) / frame_count


def init_BlobDetection(minThreshold=10,             maxThreshold=220,
                       filterByInertia=True,        minInertiaRatio=0.6,
                       filterByArea=True,           minArea=500, maxArea=np.inf,
                       filterByConvexity=True,      minConvexity=0.4,
                       filterByCircularity=True,    minCircularity=0.6,
                       filterByColor=True,          blobColor=255):
    """
    Initialize thw blob detection with default parameters
    :return: the blob detector initialized with the params
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold

    params.filterByInertia = filterByInertia
    params.minInertiaRatio = minInertiaRatio

    params.filterByArea = filterByArea
    params.minArea = minArea
    params.maxArea = maxArea

    params.filterByConvexity = filterByConvexity
    params.minConvexity = minConvexity

    params.filterByCircularity = filterByCircularity
    params.minCircularity = minCircularity

    params.filterByColor = filterByColor
    params.blobColor = blobColor

    return cv2.SimpleBlobDetector_create(params)


def find_and_bound_contours(mask, frame, areaThreshold=1500, numContours=0):
    """
    find contours on the mask and draw the bounding boxes at the corresponding positions on the frame
    :param mask: the binary mask to detect the contours on
    :param frame: the original input frame
    :param areaThreshold: the minimum area (#pixels) a successfully detected contour needs to contain
    :return: bounding rectangles of the detected contours
    """

    # TODO: areaThread should depend on the frame size
    bounding_rects = [None] * numContours
    x = [None] * numContours
    y = [None] * numContours
    w = [None] * numContours
    h = [None] * numContours
    area = [None] * numContours
    frame, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area[i] = cv2.contourArea(contour)
        if area[i] > areaThreshold:
            x[i], y[i], w[i], h[i] = cv2.boundingRect(contour) 
            bounding_rects.append([x[i], y[i], w[i], h[i]])

    return bounding_rects


def binary_color_mask_and_regulate(frame, lower, upper,
                                   erode_mask=None, dilate_mask=None, num_erode=5, num_dilate=5, second_erode=False):
    """
    generating a mask for the portions on frame that contains pixels within the range of lower and upper bounds, dilated
    and eroded for better connectivity and a more convex and regular shape
    :param frame: original input frame
    :param lower: lower bound for the colors, should have the same #channels as of frame
    :param upper: upper bound for the colors, should have the same #channels as of frame
    :param erode_mask: the kernel for eroding the raw mask, for noise reduction
    :param dilate_mask: the kernel for dilating the mask after the erosion, for better connectivity
    :param second_erode: the kernel for eroding the mask after the dilation, for removing extra padding around the
                        actual mask
    :return: mask
    """
    mask = cv2.inRange(frame, lower, upper)
    mask = cv2.erode(mask, None, iterations=num_erode)
    mask = cv2.dilate(mask, dilate_mask, iterations=num_dilate)
    if second_erode:
        mask = cv2.erode(mask, None, iterations=num_erode)
    return mask


def preprocess_frame(frame, blur_kernel_size, blur_method="average", size=(640, 480), target_colorspace="HLS"):
    """
    :param frame: the original input frame
    :param size: the target frame size
    :param blur_method: "Gaussian" or "average"
    :param blur_kernel_size: >0, <1
    :param target_colorspace: "HLS", "HSV", "RGB", etc...
    :return: preprocessed_frame: the frame that is resized, blurred, and converted
    """
    if blur_method not in ["Gaussian", "average"]:
        print("must select a blur method in", ["Gaussian", "average"])
        sys.exit(1)
    if blur_method == "average":
        blurframe = cv2.blur(frame, (int(blur_kernel_size * size[0]),
                                     int(blur_kernel_size * size[0])))
    else:
        blurframe = cv2.GaussianBlur(frame, (int(blur_kernel_size * size[0]),
                                             int(blur_kernel_size * size[0])))
    if target_colorspace not in ["HLS", "HSV", "RGB", None]:
        print("must select a color space in", ["HLS", "HSV", "RGB", None])
        sys.exit(1)
    if target_colorspace == "HLS":
        preprocessed_frame = cv2.cvtColor(blurframe, cv2.COLOR_BGR2HLS)
    elif target_colorspace == "HSV":
        preprocessed_frame = cv2.cvtColor(blurframe, cv2.COLOR_BGR2HSV)
    elif target_colorspace == "RGB":
        preprocessed_frame = cv2.cvtColor(blurframe, cv2.COLOR_BGR2RGB)
    else:
        # None is nothing - by the Zen of Python
        preprocessed_frame = blurframe
    return preprocessed_frame


def getBallonContours(detector, frame, frameContour, ratio, bcxdata, bcydata, disbdata, balloonmsg):

    balloonmsg.data = [0, 0, 0, 0]

    processed_frame = preprocess_frame(frame, mask_ROI_portion, size=frame.shape)

    blower = 0.75 * np.array([n_channel_data_b[0][0], n_channel_data_b[1][0], n_channel_data_b[2][0]])
    bupper = 1.33 * np.array([n_channel_data_b[0][2], n_channel_data_b[1][2], n_channel_data_b[2][2]])

    dilate_kernel_size = int(mask_ROI_portion * frame.shape[1])
    bmask = binary_color_mask_and_regulate(processed_frame, blower, bupper,
                                            None, np.ones((dilate_kernel_size, dilate_kernel_size)),
                                            num_erode=5, num_dilate=3,
                                            second_erode=True)

    bmask = cv2.erode(bmask, np.ones((dilate_kernel_size, dilate_kernel_size)), iterations=2)

    # Detect blobs
    keypoints = detector.detect(bmask)
    bounding_rects = find_and_bound_contours(bmask, frame, numContours=len(keypoints))
    x = [None] * len(bounding_rects)
    y = [None] * len(bounding_rects)
    w = [None] * len(bounding_rects)
    h = [None] * len(bounding_rects)
    x_avg = [None] * len(bounding_rects)
    y_avg = [None] * len(bounding_rects)
    center = [None] * len(bounding_rects)
    box_width = [None] * len(bounding_rects)
    for i, bounding_rect in enumerate(bounding_rects):
        x[i], y[i], w[i], h[i] = bounding_rect  # create bonding box
        box_width[i] = w[i]
        cv2.rectangle(frame, (x[i], y[i]), (x[i] + w[i], y[i] + h[i]), (0, 255, 0), 2)
        bcxdata[i].update(int(x[i] + w[i] / 2))
        bcydata[i].update(int(y[i] + h[i] / 2))
        x_avg[i] = int(bcxdata[i].get())
        y_avg[i] = int(bcydata[i].get())
        center[i] = (x_avg[i], y_avg[i])
        frameContour = cv2.circle(frame, center[i], radius=0, color=(0, 0, 255), thickness=10)

    if len(keypoints) > 0:
        balloonmsg.data[0] = 1
        if keypoints[0].size > 100:
            keypoints[0].size = keypoints[0].size - 20
        # Get the number of blobs found
        p = [None] * len(box_width)
        for i in len(box_width):
            p[i] = box_width[i]             # perceived width, in pixels
            w = BALLOONWIDTH         # approx. actual width, in meters (pre-computed)
            f = FOCAL_LENGTH * ratio  # camera focal length, in pixels (pre-computed)
            d[i] = f * w / p[i]
            disbdata[i].update(d)
            # cv2.putText(frame, "Distance=%.3fm" % d, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            sendDis = min(disbdata[i].get())
            index = disbdata[i].get().index(sendDis)
            balloonmsg.data[1] = int(bcxdata[index].get())
            print("Ballon Center X: ", int(bcxdata[index].get()))
            balloonmsg.data[2] = int(bcydata[index].get())
            print("Ballon Center Y: ", int(bcydata[index].get()))
            balloonmsg.data[3] = sendDis
            print("Ballon Distance=%.3fm" % sendDis)
            balloonpub.publish(balloonmsg)
            #print()

    blank = np.zeros((1, 1))

    # Draw detected blobs as red circles
    frameContour = cv2.drawKeypoints(frameContour, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return frameContour


def getShapeContours(frame, frameContour, ratio, cxdata, cydata, radiusdata, disdata, goalmsg):

    goalmsg.data = [0, 0, 0, 0]
    
    frames, contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            (sx,sy),radius = cv2.minEnclosingCircle(cnt)
            cxdata.update(int(sx))
            cydata.update(int(sy))
            radiusdata.update(int(radius))
            center = (int(sx), int(sy))
            radius = int(radiusdata.get())
            diag = 2*radius
            cv2.circle(frameContour,center,radius,(0,255,0),2)
            cv2.circle(frameContour, center, radius=0, color=(0, 0, 255), thickness=10)

            if diag:
                goalmsg.data[0] = 1
                p = diag                         # perceived width, in pixels
                w = SQUAREWIDTH                  # approx. actual width, in meters (pre-computed)
                f = FOCAL_LENGTH * ratio         # camera focal length, in pixels (pre-computed)
                d = f * w / p
                disdata.update(int(d))
                goalmsg.data[1] = int(cxdata.get())/FRAMEX
                #print("Goal X:", int(cxdata.get()))
                goalmsg.data[2] = int(cydata.get())/FRAMEY
                #print("Goal Y:", int(cydata.get()))
                goalmsg.data[3] = disdata.get()
                #print("Goal Distance=%.3fm" % disdata.get())
                goalpub.publish(goalmsg)
                #print()


            cv2.drawContours(frameContour, cnt, -1, (255, 0, 255), 7)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            if len(approx) == 3 or len(approx) == 4 or len(approx) > 15:
                if len(approx) == 3:
                    print("Triangle target")
                    cv2.drawContours(frameContour,[cnt],0,(0,255,0),3)
                elif len(approx) == 4:
                    print("Square target")
                    cv2.drawContours(frameContour,[cnt],0,(0,0,255),3)
                elif len(approx) > 15:
                    print("Circle target")
                    cv2.drawContours(frameContour,[cnt],0,(255,0,0),3)

    return frameContour


if __name__ == "__main__":
    # parse arguments
    mask_ROI_portion = 1 / 30
    mode = parse_args()

    if mode[0] == 0:
        FOCAL_LENGTH = 660
        videoCapture = cv2.VideoCapture(0)
        if not videoCapture.isOpened():
            print("Failed to open picam!!!")
            sys.exit()
        
    else:

        FOCAL_LENGTH = 1460
        videoCapture = cv2.VideoCapture(0)
        if not videoCapture.isOpened():
            print("Failed to open cvcam!!!")
            sys.exit()

    """
    This mode allows the ballon and the goal to be detected simultaneously
    """
    if mode[1] == 0:

        # obtain the calibration file for the ballon
        try:
            with open("balloncolorinfo.dat", "rb") as file1:
                color = pickle.load(file1)
                n_channel_data_b = pickle.load(file1)
        except FileNotFoundError:
            print("Calibrate the color first!!!")
            sys.exit()
        # obtain the calibration file for the goal
        try:
            with open("goalcolorinfo.dat", "rb") as file2:
                color = pickle.load(file2)
                n_channel_data_g = pickle.load(file2)
        except FileNotFoundError:
            print("Calibrate the color first!!!")
            sys.exit()

        # allow the camera to warmup
        ret, frame = videoCapture.read()
        rate.sleep()
       
        # X and Y for the center of the ballon
        bcx_data = [None] * ([0] * 15)
        bcxdata = TrackingDetection(bcx_data)
        bcy_data = [None] * ([0] * 15)
        bcydata = TrackingDetection(bcy_data)
        # Distance of the ballon
        disb_data = [None] * ([0] * 15)
        disbdata = TrackingDetection(disb_data)

        # X and Y for the center of the goal
        cx_data = [0] * 15
        cxdata = TrackingDetection(cx_data)
        cy_data = [0] * 15
        cydata = TrackingDetection(cy_data)
        # radius
        radius_data = [0] * 15
        radiusdata = TrackingDetection(radius_data)
        # Distance of the goal
        dis_data = [0] * 15
        disdata = TrackingDetection(dis_data)

        while not rospy.is_shutdown():
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            ret, frame = videoCapture.read()
            rate.sleep()
            if not ret:
                print("Frame capture failed!!!")
                break
            # get the desired frame size
            ratio = 600 / frame.shape[1]
            dim = (600, int(frame.shape[0] * ratio))
            FRAMEX = 600
            FRAMEY =  int(frame.shape[0] * ratio)
            # resize the image
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
            frameContour = frame.copy()
            
            # send frame dimensions
            #framemsg = Float64MultiArray()
            #framemsg.data = dim
            #framepub.publish(framemsg)

            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
            blurFrame = cv2.GaussianBlur(frame, (7, 7), 1)
            hsvframe = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
            kernel = np.ones((3, 3), dtype=np.uint8)
            
            glower = 0.75 * np.array([n_channel_data_g[0][0], n_channel_data_g[1][0], n_channel_data_g[2][0]])
            gupper = 1.33 * np.array([n_channel_data_g[0][2], n_channel_data_g[1][2], n_channel_data_g[2][2]])
            gmask = cv2.inRange(hsvframe, glower, gupper)

            kernel = np.ones((7, 7), dtype=np.uint8)
            colorFrame = cv2.bitwise_and(frame, frame, mask=gmask)
            dilateFrame = cv2.dilate(gmask, kernel, iterations=0)

            balloonmsg = Float64MultiArray()
            goalmsg = Float64MultiArray()

            # detect the goal
            #goalTargetFrame = getShapeContours(dilateFrame, frameContour, ratio, cxdata, cydata, radiusdata, disdata, goalmsg)
            #cv2.imshow("goalTargetFrame", goalTargetFrame)

            # detect the ballon
            detector = init_BlobDetection()
            ballonTargetFrame = getBallonContours(detector, frame, frameContour, ratio, bcxdata, bcydata, disbdata, balloonmsg)
            #cv2.imshow("ballonTargetFrame", ballonTargetFrame)
            
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

            if cv2.waitKey(33) == 27:
                # De-allocate any associated memory usage
                cv2.destroyAllWindows()
                videoCapture.release()
                break



    # perform calibration or detection 
    elif mode[1] == 1:
        # calibration mode, calibrated file will be stored in a colorinfo.dat file
        color_space = "HLS"
        hsl_channel_data_array = [[255, 0, 0] for _ in range(3)]

        time_begin = time.time()
        frame_count = 0
        videoCapture.read()
        rate.sleep()
        while time.time() - time_begin < 5:
            frame_count += 1
            ret, frame = videoCapture.read()
            if not ret:
                print("Frame capture failed!!!")
                break
            else:
                crop_frame = \
                    frame[int(frame.shape[0] * (1 / 2 - mask_ROI_portion / 2)):int(
                        frame.shape[0] * (1 / 2 + mask_ROI_portion / 2)
                    ), int(frame.shape[1] * (1 / 2 - mask_ROI_portion / 2)):int(
                        frame.shape[1] * (1 / 2 + mask_ROI_portion / 2)
                    )]
                frame_hls = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HLS)
                n_channel_min_mean_max(frame_hls, frame_count, hsl_channel_data_array)
                ### DEBUG below
                cv2.imshow("frame", cv2.cvtColor(frame_hls, cv2.COLOR_HLS2BGR))
                cv2.waitKey(1)

        # Destroying All the windows
        cv2.destroyAllWindows()

        # using the wait key function to delay
        # the closing of windows till any key is pressed
        with open("balloncolorinfo.dat", "wb") as file:
            pickle.dump(color_space, file)
            pickle.dump(hsl_channel_data_array, file)
        print("Calibration complete, balloncolorinfo.dat file generated.")
        sys.exit()


    elif mode[1] == 2:
        # detection mode using color filter
        # obtain the calibration file
        try:
            with open("balloncolorinfo.dat", "rb") as file:
                color = pickle.load(file)
                n_channel_data = pickle.load(file)
        except FileNotFoundError:
            print("Calibrate the color first!!!")
            sys.exit()

        # allow the camera to warmup
        ret, frame = videoCapture.read()
        rate.sleep()

        # capture frames from the camera
        while True:
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            ret, frame = videoCapture.read()

            ratio = 600 / frame.shape[1]
            dim = (600, int(frame.shape[0] * ratio))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

            if not ret:
                print("Frame capture failed!!!")
                break

            processed_frame = preprocess_frame(frame, mask_ROI_portion, size=frame.shape)

            lower = np.array([n_channel_data[0][0], n_channel_data[1][0], n_channel_data[2][0]])
            upper = np.array([n_channel_data[0][2], n_channel_data[1][2], n_channel_data[2][2]])

            dilate_kernel_size = int(mask_ROI_portion * frame.shape[1])
            mask = binary_color_mask_and_regulate(processed_frame, lower, upper,
                                                  None, np.ones((dilate_kernel_size, dilate_kernel_size)),
                                                  num_erode=5, num_dilate=3,
                                                  second_erode=True)

            bounding_rects = find_and_bound_contours(mask, frame)
            for bounding_rect in bounding_rects:
                x, y, w, h = bounding_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Target", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            # show the frame
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

            if cv2.waitKey(33) == 27:
                # De-allocate any associated memory usage
                cv2.destroyAllWindows()
                videoCapture.release()
                break


    elif mode[1] == 3:
        # detection mode using blob detection
        # obtain the calibration file
        try:
            with open("balloncolorinfo.dat", "rb") as file:
                color = pickle.load(file)
                n_channel_data = pickle.load(file)
                # print("Color space: ", color)
                # print("Color data: ", n_channel_data)
        except FileNotFoundError:
            print("Calibrate the color first!!!")
            sys.exit()

        # Constant for focal length, in pixels (must be changed per camera)
        # TODO: automate this using the tutorial I sent

        detector = init_BlobDetection()

        # allow the camera to warmup
        ret, frame = videoCapture.read()
        rate.sleep()
        # capture frames from the camera

        cx_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        cxdata = TrackingDetection(cx_data)
        cy_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        cydata = TrackingDetection(cy_data)

        while True:
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            ret, frame = videoCapture.read()

            ratio = 600 / frame.shape[1]
            dim = (600, int(frame.shape[0] * ratio))

            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

            if not ret:
                print("Frame capture failed!!!")
                break

            processed_frame = preprocess_frame(frame, mask_ROI_portion, size=frame.shape)

            lower = 0.75 * np.array([n_channel_data[0][0], n_channel_data[1][0], n_channel_data[2][0]])
            upper = 1.33 * np.array([n_channel_data[0][2], n_channel_data[1][2], n_channel_data[2][2]])

            dilate_kernel_size = int(mask_ROI_portion * frame.shape[1])
            mask = binary_color_mask_and_regulate(processed_frame, lower, upper,
                                                  None, np.ones((dilate_kernel_size, dilate_kernel_size)),
                                                  num_erode=5, num_dilate=3,
                                                  second_erode=True)

            mask = cv2.erode(mask, np.ones((dilate_kernel_size, dilate_kernel_size)), iterations=2)

            box_width = 0
            bounding_rects = find_and_bound_contours(mask, frame)
            for bounding_rect in bounding_rects:
                x, y, w, h = bounding_rect  # create bonding box
                box_width = w
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cxdata.update(int(x + w / 2))
                cydata.update(int(y + h / 2))
                x_avg = int(cxdata.get())
                y_avg = int(cydata.get())
                center = (x_avg, y_avg)
                frame = cv2.circle(frame, center, radius=0, color=(0, 0, 255), thickness=10)

            # Detect blobs
            keypoints = detector.detect(mask)

            if len(keypoints) > 0:
                if keypoints[0].size > 100:
                    keypoints[0].size = keypoints[0].size - 30
                # Get the number of blobs found
                blobCount = len(keypoints)
                if box_width:
                    p = box_width             # perceived width, in pixels
                    w = BALLOONWIDTH          # approx. actual width, in meters (pre-computed)
                    f = FOCAL_LENGTH * ratio  # camera focal length, in pixels (pre-computed)
                    d = f * w / p
                    # cv2.putText(frame, "Distance=%.3fm" % d, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    print("Ballon Distance=%.3fm" % d)

            blank = np.zeros((1, 1))

            # Draw detected blobs as red circles
            blobs = cv2.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Show the frames
            # frameStack = stackFrames(1, [mask, blobs])
            cv2.imshow('Mask', mask)
            cv2.imshow('Blobs', blobs)

            if cv2.waitKey(33) == 27:
                # De-allocate any associated memory usage
                cv2.destroyAllWindows()
                videoCapture.release()
                break


    elif mode[1] == 4:
        # calibration mode, calibrated file will be stored in a colorinfo.dat file
        color_space = "HLS"
        hsl_channel_data_array = [[255, 0, 0] for _ in range(3)]

        time_begin = time.time()
        frame_count = 0
        videoCapture.read()
        rate.sleep()
        while time.time() - time_begin < 5:
            frame_count += 1
            ret, frame = videoCapture.read()
            if not ret:
                print("Frame capture failed!!!")
                break
            else:
                crop_frame = \
                    frame[int(frame.shape[0] * (1 / 2 - mask_ROI_portion / 2)):int(
                        frame.shape[0] * (1 / 2 + mask_ROI_portion / 2)
                    ), int(frame.shape[1] * (1 / 2 - mask_ROI_portion / 2)):int(
                        frame.shape[1] * (1 / 2 + mask_ROI_portion / 2)
                    )]
                frame_hls = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HLS)
                n_channel_min_mean_max(frame_hls, frame_count, hsl_channel_data_array)
                ### DEBUG below
                cv2.imshow("frame", cv2.cvtColor(frame_hls, cv2.COLOR_HLS2BGR))
                cv2.waitKey(1)

        # Destroying All the windows
        cv2.destroyAllWindows()

        # using the wait key function to delay
        # the closing of windows till any key is pressed
        with open("goalcolorinfo.dat", "wb") as file:
            pickle.dump(color_space, file)
            pickle.dump(hsl_channel_data_array, file)
        print("Calibration complete, goalcolorinfo.dat file generated.")
        sys.exit()


    elif mode[1] == 5:
        def nothing(x):
            pass

        cv2.namedWindow('Parameters')
        # creating trackbars threshold 1
        cv2.createTrackbar('Threshold 1', 'Parameters', 150, 255, nothing)
        # creating trackbars threshold 2
        cv2.createTrackbar('Threshold 2', 'Parameters', 200, 255, nothing)

        # allow the camera to warmup
        ret, frame = videoCapture.read()
        rate.sleep()

        # capture frames from the camera
        while True:
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            ret, frame = videoCapture.read()

            ratio = 600 / frame.shape[1]
            dim = (600, int(frame.shape[0] * ratio))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
            frameContour = frame.copy()
            
            if not ret:
                print("Frame capture failed!!!")
                break

            blurFrame = cv2.GaussianBlur(frame, (19, 19), 1)
            grayFrame = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2GRAY)

            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
            cannyFrame = cv2.Canny(grayFrame, threshold1, threshold2)
            
            kernel = np.ones((5, 5))
            dilateFrame = cv2.dilate(cannyFrame, kernel, iterations=1)

            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (101,1))
            #morph = cv2.morphologyEx(dilateFrame, cv2.MORPH_CLOSE, kernel)

            #cv2.imshow("morph", morph)

            targetFrame = getShapeContours(dilateFrame, frameContour, ratio)

            cv2.imshow('Canny', cannyFrame)
            cv2.imshow('Dilate', dilateFrame)
            cv2.imshow('Target', targetFrame)

            if cv2.waitKey(33) == 27:
                # De-allocate any associated memory usage
                cv2.destroyAllWindows()
                videoCapture.release()
                break


    elif mode[1] == 6:
        # Ballon detection mode using color filter
        # obtain the calibration file
        try:
            with open("goalcolorinfo.dat", "rb") as file:
                color = pickle.load(file)
                n_channel_data = pickle.load(file)
        except FileNotFoundError:
            print("Calibrate the color first!!!")
            sys.exit()

        # allow the camera to warmup
        ret, frame = videoCapture.read()
        rate.sleep()

        # X and Y for the center of the goal
        cx_data = [0] * 15
        cxdata = TrackingDetection(cx_data)
        cy_data = [0] * 15
        cydata = TrackingDetection(cy_data)
        # radius
        radius_data = [0] * 15
        radiusdata = TrackingDetection(radius_data)
        # Distance in meters
        dis_data = [0] * 15
        disdata = TrackingDetection(dis_data)

        # capture frames from the camera
        while True:
            ret, frame = videoCapture.read()
            ratio = 600 / frame.shape[1]
            dim = (600, int(frame.shape[0] * ratio))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
            frameContour = frame.copy()
            if not ret:
                print("Frame capture failed!!!")
                break

            blurFrame = cv2.GaussianBlur(frame, (9, 9), 1)
            hsvframe = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
            
            lower = 0.75 * np.array([n_channel_data_g[0][0], n_channel_data_g[1][0], n_channel_data_g[2][0]])
            upper = 1.33 * np.array([n_channel_data_g[0][2], n_channel_data_g[1][2], n_channel_data_g[2][2]])
            mask = cv2.inRange(hsvframe, lower, upper)
            #cv2.imshow('Mask', mask)
            #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            #cv2.imshow('After', mask)

            kernel = np.ones((5, 5), dtype=np.uint8)
            colorFrame = cv2.bitwise_and(frame, frame, mask=mask)
            #erodeFrame = cv2.erode(mask, kernel, iterations=1)
            dilateFrame = cv2.dilate(mask, kernel, iterations=1)

            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (101,1))
            #morph = cv2.morphologyEx(dilateFrame, cv2.MORPH_CLOSE, kernel)
            #cv2.imshow('Morph', morph)

            targetFrame = getShapeContours(dilateFrame, frameContour, ratio, cxdata, cydata, radiusdata, disdata)

            cv2.imshow('Color', colorFrame)
            cv2.imshow('Dilate', dilateFrame)
            cv2.imshow('Target', targetFrame)

            if cv2.waitKey(33) == 27:
                # De-allocate any associated memory usage
                cv2.destroyAllWindows()
                videoCapture.release()
                break
        

    else:
        print("Invalid mode!!!")
        sys.exit()
