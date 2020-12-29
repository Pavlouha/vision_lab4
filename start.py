from PIL import Image
import cv2
import torch
from numpy import mat
from torchvision.models import alexnet

from utils import read_classes, processing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture('traffic.mp4')

ret, frame = cap.read()
print('Video params: ret =', ret, 'W =', frame.shape[1], 'H =', frame.shape[0], 'channel =', frame.shape[2])

FPS = 25.0
FrameSize = (frame.shape[1], frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('output.avi', fourcc, FPS, FrameSize, 0)
#out = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'H264'), 1, FrameSize)

model = alexnet(pretrained=True).eval().cuda()

classes = read_classes()

while (cap.isOpened()):
    ret, frame = cap.read()

    # check for successfulness of cap.read()
    if not ret: break

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = processing(pil_img, model, device, classes)

    # Save the video
    # out.write(frame)
    out.write(img)

    # cv2.imshow('frame', frame)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
