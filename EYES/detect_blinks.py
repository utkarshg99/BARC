#Eye blink detection with OpenCV, Python, and dlibPython
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3
blink_rate = np.array([])
time_array = np.array([])
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
init_time = time.time()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# start the video stream thread
print("[INFO] starting video stream thread...")
#FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=1).start()
pupil_area = np.array([])
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	#if fileStream and not vs.more():
	#	break
 
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	#print(frame)
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 	faces = cv2.CascadeClassifier('haarcascade_eye.xml')
	# detect faces in the grayscale frame
	new_frame = gray[:,0:320]
	rects = detector(gray, 0)
	# loop over the face detections
	detected = faces.detectMultiScale(new_frame, 1.3, 5)
	pupilFrame = new_frame
	pupilO = new_frame
	windowClose = np.ones((5,5),np.uint8)
	windowOpen = np.ones((2,2),np.uint8)
	windowErode = np.ones((2,2),np.uint8)
	for (x,y,w,h) in detected:
		cv2.rectangle(new_frame, (x,y), ((x+w),(y+h)), (0,0,255),1)	
		cv2.line(new_frame, (x,y), ((x+w,y+h)), (0,0,255),1)
		cv2.line(new_frame, (x+w,y), ((x,y+h)), (0,0,255),1)
		pupilFrame = cv2.equalizeHist(new_frame[np.int((y)+(h)*.25):y+h, x:x+w])
		pupilO = pupilFrame
		ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)		#50 ..nothin 70 is better
		pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
		pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
		pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
		threshold = cv2.inRange(pupilFrame,252,255)		#get the blobs
		_,contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
 
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
 
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
 
			# reset the eye frame counter
			COUNTER = 0
			# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		time_elapsed = time.time() - init_time
		cv2.putText(frame, "Blink Rate: {}".format(TOTAL/time_elapsed*60), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		blink_rate = np.append(blink_rate, (TOTAL/time_elapsed)*60)
		time_array = np.append(time_array, time_elapsed)
 	#(TOTAL/time_elapsed)*60
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
plt.plot(blink_rate)
plt.show()
