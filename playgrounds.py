import cv2
import torch
from PIL import Image
from torchvision.models import alexnet

from utils import read_classes, predict_image

if __name__ == '__main__':

    # Различные трекеры - из стаковерфлоу
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
    tracker_type = tracker_types[6]
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()

    # Достаём модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alexnet(pretrained=True).eval().cuda()
    classes = read_classes()

    # параметр 0 для видевокамеры, 'traffic.mp4' для видевы
    video = cv2.VideoCapture(0)

    # Читаем первый кадр
    ok, frame = video.read()

    # x, y, w, h - контур кодом
    #bbox = (287, 23, 86, 320)

    #будем кадры считать
    index = 0

    # Выбираем собственный контур мышкой - первый y, второй - икс
    bbox = cv2.selectROI(frame, False)

    #cv2.imwrite("roi.png", roi)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        if index==0:
            roi = frame[int(bbox[1]):int(bbox[1]) + int(bbox[2]), int(bbox[0]):int(bbox[0]) + int(bbox[3])]
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predicted = predict_image(pil_img, model, device, classes)

        if index==50:
            roi = frame[int(bbox[1]):int(bbox[1]) + int(bbox[2]), int(bbox[0]):int(bbox[0]) + int(bbox[3])]
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predicted = predict_image(pil_img, model, device, classes)
            index=0

        # Calculate FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        #cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, 'Class: '+predicted, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # ФПС
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Результат
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
