import math
import cv2
import numpy as np
import glob
import pickle
import time

NUM_SAMPLES = 30

def calibrate(n = NUM_SAMPLES, debug=False):
	# some criteria for calibration cycles
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# storage for detected points
	objp = np.zeros((7*7, 3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
	objpoints = []
	imgpoints = []

	img_size = None

	print("Begin chessboard exercise!")

	# start video capture
	cap = cv2.VideoCapture(1)
	if not cap.isOpened():
		print("No camera device found")
		exit()

	# loop until enough samples are collected
	while len(objpoints) < NUM_SAMPLES:
		# get image
		ret, img = cap.read()

		if not ret:
			print("Could not take picture")
			exit()
		
		if img_size == None:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img_size = gray.shape[::-1]
		
		# convert to grayscale
		res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# show image
		if debug:
			cv2.imshow('Chessboard', res)
			cv2.waitKey(100)
		
		# find corners (if possible)
		ret, corners = cv2.findChessboardCorners(res, (7,7), None)
		if ret == True:
			print("Success! Adjusting and displaying...")
			objpoints.append(objp)
			corners2 = cv2.cornerSubPix(res, corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners)
			if debug:
				cv2.drawChessboardCorners(res, (7,7), corners2, ret)
				cv2.imshow('Chessboard', res)
				cv2.waitKey(100)
			else:
				time.sleep(0.1)
		else:
			#print("Something went wrong! Skipping...")
			pass

	print("\nFound %d object points and %d image points." % (len(objpoints), len(imgpoints)))

	# compute camera matrix
	ret, mat, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, cameraMatrix = None, distCoeffs = None)
	print("Matrix:")
	print(mat)

	cv2.destroyAllWindows()

	# save matrix
	with open("camera_matrix.dat", "wb") as f:
		pickle.dump(mat, f)

if __name__ == '__main__':
    calibrate(debug=False)