from ultralytics import YOLO
import cv2
import random


def videoDetection(yolo_model, videofile, threshold):

    cap = cv2.VideoCapture(videofile)
    pre_frame_time, new_frame_time = 0, 0

    while (cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break

        output = yolo_model(frame)[0]

        boxes = output.boxes.xyxy
        class_ids = output.boxes.cls
        scores = output.boxes.conf

        valid_idx = scores > threshold
        boxes = boxes[valid_idx]
        class_ids = class_ids[valid_idx]
        scores = scores[valid_idx]

        colors = [(256,0, 0), (0,0,0), (0,256,0),(0,0,256)]
        random.shuffle(colors)
        colors = [colors[int(i%len(colors))] for i in class_ids]

        cls_names = [output.names[int(x)] for x in class_ids]
        labels = [f"{cls} {round(s.numpy()*100)}%"
                    for cls, s in zip(cls_names, scores)]

        for i in range(len(scores)):
            (x, y) = (int(boxes[i][0]), boxes[i][1])
            (x2, y2) = (boxes[i][2], boxes[i][3])
            color = colors[i]
            cv2.rectangle(frame, (int(x), int(y)),(int(x2), int(y2)), color,2)
            cv2.putText(frame, labels[i], (int(x), int(y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

        cv2.imshow('new_frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float,
                        default=.8)
    parser.add_argument('-f', '--video', default='./data/demoVideo.mp4')

    args = parser.parse_args()

    yolo = yolo = YOLO('yolov8n.pt')

    videoDetection(yolo, args.video, args.threshold)
