# import the necessary packages
import time
import cv2
import numpy as np
import sys
import pickle

def parse_args():
    """Function to parse the system arguments"""
    # mode[0] defines frame obtaining methods - 0: picamera; 1: opencv camera
    # mode[1] defines operation mode - 0: detection mode; 1: calibration mode
    mode = [1, 0]
    if len(sys.argv) < 2:
        print("ERROR: Insufficient arguments",
              "Please use the following form:",
              "python main.py <picam/cvcam> <1 if calibration mode else leave blank or input anything to nullify>")
        sys.exit()
    elif len(sys.argv) > 3:
        print("ERROR: Too many arguments",
              "Please use the following form:",
              "python main.py <picam/cvcam> <1 if calibration mode else leave blank or input anything to nullify>")
        sys.exit()
    else:
        if sys.argv[1] == "picam":
            print("picamera in progress, exiting!")
            # TODO: fix picamera
            # from picamera.array import PiRGBArray
            # from picamera import PiCamera
            # camera = PiCamera()
            # camera.resolution = RES
            # camera.framerate = 25
            # rawCapture = PiRGBArray(camera, size=RES)
            mode[0] = 0
            sys.exit()
        else:
            assert sys.argv[1] == "cvcam"
            mode[0] = 1
            if len(sys.argv) == 3:
                if int(sys.argv[2]) == 1:
                    mode[1] = 1
                    print("Color calibration mode, please place the balloon in the center of the frame until the program exits")
                else:
                    print("Balloon detection mode.")
            else:
                print("Balloon detection mode.")
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
    data_array[1] = data_array[1] + (mean_val - data_array[1])/frame_count


if __name__== "__main__":
    # parse arguments
    mask_ROI_portion = 1 / 20
    mode = parse_args()

    if mode[0] == 0:
        # TODO: picamera imports
        pass
    else:
        videoCapture = cv2.VideoCapture(0)
        if not videoCapture.isOpened():
            print("Failed to open cvcam!!!")
            sys.exit()

    if mode[1] == 1:
        # calibration mode, calibrated file will be stored in a colorinfo.dat file
        color_space = "HLS"
        hsl_channel_data_array = [[255, 0, 0] for _ in range(3)]

        time_begin = time.time()
        frame_count = 0
        while time.time() - time_begin < 5:
            frame_count += 1
            ret, frame = videoCapture.read()
            if not ret:
                print("Frame capture failed!!!")
                break
            else:
                crop_frame = \
                    frame[int(frame.shape[0]*(1/2 - mask_ROI_portion/2)):int(frame.shape[0]*(1/2 + mask_ROI_portion/2)),
                          int(frame.shape[1]*(1/2 - mask_ROI_portion/2)):int(frame.shape[1]*(1/2 + mask_ROI_portion/2))]
                frame_hls = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HLS)
                n_channel_min_mean_max(frame_hls, frame_count, hsl_channel_data_array)
                ### DEBUG below
                cv2.imshow("frame", cv2.cvtColor(frame_hls, cv2.COLOR_HLS2BGR))
                cv2.waitKey(1)

        # Destroying All the windows
        cv2.destroyAllWindows()

        # using the wait key function to delay
        # the closing of windows till any key is pressed
        with open("colorinfo.dat", "wb") as file:
            pickle.dump(color_space, file)
            pickle.dump(hsl_channel_data_array, file)
    else:
        # detection mode
        # obtain the calibration file
        try:
            with open("colorinfo.dat", "rb") as file:
                color = pickle.load(file)
                n_channel_data = pickle.load(file)
        except FileNotFoundError:
            print("Calibrate the color first!!!")
            sys.exit()

        # TODO: compare to blob detection
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 10
        params.maxThreshold = 220

        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        params.filterByArea = True
        params.minArea = 500
        params.maxArea = np.inf

        params.filterByConvexity = True
        params.minConvexity = 0.3

        params.filterByCircularity = True
        params.minCircularity = 0.6

        params.filterByColor = True
        params.blobColor = 255

        # allow the camera to warmup
        time.sleep(0.1)

        # capture frames from the camera
        while True:
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            ret, frame = videoCapture.read()
            if not ret:
                print("Frame capture failed!!!")
                break

            blurframe = cv2.blur(frame, (int(mask_ROI_portion*frame.shape[1]), int(mask_ROI_portion*frame.shape[1])))
            hlsframe = cv2.cvtColor(blurframe, cv2.COLOR_BGR2HLS)

            lower = np.array([n_channel_data[0][0], n_channel_data[1][0], n_channel_data[2][0]])
            upper = np.array([n_channel_data[0][2], n_channel_data[1][2], n_channel_data[2][2]])

            mask = cv2.inRange(hlsframe, lower, upper)
            mask = cv2.erode(mask, None, iterations=5)
            mask = cv2.dilate(mask, np.ones((int(mask_ROI_portion*frame.shape[1]), int(mask_ROI_portion*frame.shape[1]))), iterations=3)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 2500:
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "Target"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, text, (x, y), font, 1, (0, 255, 0))
            # show the frame
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            cv2.waitKey(1)