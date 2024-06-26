import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from model import *

def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)

    probs = torch.nn.functional.softmax(output, dim=1)
    prob=torch.Tensor.tolist(probs)
    prob1="{0:.2f}".format(prob[0][0]*100)
    prob2="{0:.2f}".format(prob[0][1]*100)

    class_names = ['bike', 'car']
    prediction = [class_names[predicted.item()],prob1,prob2]
    return prediction

