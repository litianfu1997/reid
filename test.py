import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture('img/TownCentreXVID.avi')
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        if ret:
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", frame)
            if cv2.waitKey(1) == ord('q'):
                break
