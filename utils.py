import csv

from torch.autograd import Variable
from torchvision import transforms

# Transforms image for model input
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      ])


def read_classes():
    # Read classes from text doc
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
    return str(classes[index])
