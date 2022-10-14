# import the necessary packages
import time
import cv2
import numpy as np
import sys
import pickle
from TrackingDetection import TrackingDetection


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
            mode[0] = 0
            sys.exit()
        else:
            assert sys.argv[1] == "cvcam"
            mode[0] = 1
            if len(sys.argv) == 3:
                if int(sys.argv[2]) == 1:
                    mode[1] = 1
                    print("Color calibration mode, please place the balloon in the center of the frame")
                elif int(sys.argv[2]) == 2:
                    print("Balloon detection color filter mode.")
                    mode[1] = 2
                elif int(sys.argv[2]) == 3:
                    print("Balloon detection blob detection mode.")
                    mode[1] = 3
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


def find_and_bound_contours(mask, frame, areaThreshold=1500):
    """
    find contours on the mask and draw the bounding boxes at the corresponding positions on the frame
    :param mask: the binary mask to detect the contours on
    :param frame: the original input frame
    :param areaThreshold: the minimum area (#pixels) a successfully detected contour needs to contain
    :return: bounding rectangles of the detected contours
    """

    # TODO: areaThread should depend on the frame size
    bounding_rects = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > areaThreshold:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rects.append([x, y, w, h])

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


if __name__ == "__main__":
    # parse arguments
    mask_ROI_portion = 1 / 20
    mode = parse_args()

    # initialize the pi camera or web camera and grab a reference to the raw camera capture
    if mode[0] == 0:
        # TODO: picamera imports
        # import the necessary packages
        from picamera.array import PiRGBArray
        from queue import Queue
        from picamera import PiCamera

        FOCAL_LENGTH = 640
        BALLOON_WIDTH = 0.33

        RES = (640, 480) # esolution
        camera = PiCamera()
        camera.resolution = RES
        camera.framerate = 30
        rawCapture = PiRGBArray(camera, size=RES)
        # Constant for focal length, in pixels
        FOCAL_LENGTH = 630
        # Costant for ballon width, in meters
        BALLOON_WIDTH = 0.33

        time.sleep(0.1)
        print("picamera support incoming")
        sys.exit()
        pass

    else:
        FOCAL_LENGTH = 1460
        BALLOON_WIDTH = 0.33
        videoCapture = cv2.VideoCapture(0)
        if not videoCapture.isOpened():
            print("Failed to open cvcam!!!")
            sys.exit()

    # perform calibration or detection 
    if mode[1] == 1:
        # calibration mode, calibrated file will be stored in a colorinfo.dat file
        color_space = "HLS"
        hsl_channel_data_array = [[255, 0, 0] for _ in range(3)]

        time_begin = time.time()
        frame_count = 0
        videoCapture.read()
        time.sleep(1)
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
        with open("colorinfo.dat", "wb") as file:
            pickle.dump(color_space, file)
            pickle.dump(hsl_channel_data_array, file)
        print("Calibration complete, colorinfo.dat file generated.")
        sys.exit()

    elif mode[1] == 2:
        # detection mode using color filter
        # obtain the calibration file
        try:
            with open("colorinfo.dat", "rb") as file:
                color = pickle.load(file)
                n_channel_data = pickle.load(file)
        except FileNotFoundError:
            print("Calibrate the color first!!!")
            sys.exit()

        # allow the camera to warmup
        ret, frame = videoCapture.read()
        time.sleep(0.1)

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
            with open("colorinfo.dat", "rb") as file:
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
        time.sleep(0.1)
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

            lower = 0.75 * np.array([n_channel_data[0][0], n_channel_data[1][0], n_channel_data[2][0]])
            upper = 1.33 * np.array([n_channel_data[0][2], n_channel_data[1][2], n_channel_data[2][2]])

            dilate_kernel_size = int(mask_ROI_portion * frame.shape[1])
            mask = binary_color_mask_and_regulate(processed_frame, lower, upper,
                                                  None, np.ones((dilate_kernel_size, dilate_kernel_size)),
                                                  num_erode=5, num_dilate=3,
                                                  second_erode=True)

            box_width = 0
            bounding_rects = find_and_bound_contours(mask, frame)
            for bounding_rect in bounding_rects:
                x, y, w, h = bounding_rect  # create bonding box
                box_width = w
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center = (int(x + w / 2), int(y + h / 2))
                frame = cv2.circle(frame, center, radius=0, color=(0, 0, 255), thickness=10)

            # Detect blobs
            keypoints = detector.detect(mask)

            if len(keypoints) > 0:
                print(keypoints[0].size)
                if keypoints[0].size > 100:
                    keypoints[0].size = keypoints[0].size - 30
                # Get the number of blobs found
                blobCount = len(keypoints)
                print(blobCount, "found")
                if box_width:
                    p = box_width  # perceived width, in pixels
                    w = BALLOON_WIDTH  # approx. actual width, in meters (pre-computed)
                    f = FOCAL_LENGTH  # camera focal length, in pixels (pre-computed)
                    d = f * w / p
                    cv2.putText(frame, "Distance=%.3fm" % d, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            blank = np.zeros((1, 1))

            # Draw detected blobs as red circles
            blobs = cv2.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imshow('Blobs', blobs)
            cv2.imshow('Mask', mask)

            if cv2.waitKey(33) == 27:
                # De-allocate any associated memory usage
                cv2.destroyAllWindows()
                videoCapture.release()
                break

    else:
        print("Invalid mode!!!")
        sys.exit()
