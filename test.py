from detector import Detector
import cv2


if __name__ == '__main__':
    # 实例化目标检测模型
    detector = Detector()
    cap = cv2.VideoCapture('img/TownCentreXVID.avi')
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
        class_list = ['person']
        if ret:
            bboxes = detector.detect(frame, class_list)
            if len(bboxes) > 0:
                count = 0
                for x1, y1, x2, y2, lbl, conf in bboxes:
                    count += 1
                    color = (0, 255, 0)
                    # 裁剪各个行人
                    # ximg = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", frame)
            if cv2.waitKey(1) == ord('q'):
                break
