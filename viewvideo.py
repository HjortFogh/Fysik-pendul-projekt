import cv2

video_path = "videos/forsøg1a.mov"
capture = cv2.VideoCapture(video_path)

cv2.namedWindow("main", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("main", 1080, 1920)

while True:
    is_available, frame = capture.read()
    if not is_available: break
    cv2.imshow("main", frame)
    if cv2.waitKey(1) == ord("q"): break

capture.release()
cv2.destroyAllWindows()