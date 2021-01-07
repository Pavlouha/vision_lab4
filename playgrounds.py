import cv2
import torch
from PIL import Image
from torchvision.models import alexnet

from utils import read_classes, predict_image

if __name__ == '__main__':

    # Trackers (from StackOverflow)
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
    tracker_type = tracker_types[5]
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

    # Open our model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alexnet(pretrained=True).eval().cuda()
    classes = read_classes()

    # параметр 0 для камеры, 'traffic.mp4' для видео
    video = cv2.VideoCapture('banana_fhd.mp4')

    # read 1st frame
    ok, frame = video.read()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = frame.shape[1]
    vh = frame.shape[0]
    print("Video size", vw, vh)
    outvideo = cv2.VideoWriter("out.mp4", fourcc, 30.0, (vw, vh))

    # frame counter
    index = 0

    # x, y, w, h - контур кодом
    # bbox = (287, 23, 86, 320)

    # Выбираем собственный контур мышкой y,x
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker
    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)

        # Calculate FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            if index == 0:
                roi = frame[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])]
                pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                #cv2.imwrite("roi.png", roi)
                predicted = predict_image(pil_img, model, device, classes)

            if index == 50:
                roi = frame[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])]
                pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                predicted = predict_image(pil_img, model, device, classes)
                index = 1

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        # cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, 'Class: ' + predicted, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # fps
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        index = index + 1
        # Result
        # cv2.resizeWindow("Tracking",800, 600)
        cv2.imshow("Tracking", frame)
        outvideo.write(frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

    cv2.destroyAllWindows()
    outvideo.release()
