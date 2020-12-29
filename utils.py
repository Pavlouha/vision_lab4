import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm

# Transforms image for model input
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      ])


def read_classes():
    # Читаем классы из текстового документа
    classes = []
    with open('imagenet.txt', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            classes.append(row)
    return classes


def predict_image(image, model, device, classes):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    #return index
    return str(classes[index])


def processing(image, model, device, classes):
    fig = plt.figure()
    sub = fig.add_subplot(1, 1, 1)
    index = predict_image(image, model, device, classes)
    sub.set_title("class " + str(classes[index]))
    plt.axis('off')
    plt.imshow(image, cmap=cm.get_cmap('Greys_r'))
    # put pixel buffer in numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    # plt.savefig('./output/' + str(index) + '.jpg')
    # plt.show()
    return mat
