import cv2
import imutils
from imutils import contours

# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
refImagePath = "./image/ocr_a.jpg"
ref = cv2.imread(refImagePath)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
# ref = cv2.GaussianBlur(ref, (5,5), 0)
ref = cv2.medianBlur(ref, 5)
ref = cv2.threshold(ref, 50, 255, cv2.THRESH_BINARY)[1]
ref = ref[:, :-5]
cv2.imwrite("./image/binaryRef.jpg", ref)
# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))
	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi
	cv2.imwrite(f"./image/roi/roi_{i}.jpg", roi)