import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load('new_model_florida_plate.pth'))
model = model.to(device)

model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


image_path = 'newyork.webp' #image to test the model.
image = Image.open(image_path)
image = transform(image).unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

class_names = ['Florida', 'Non-Florida']

predicted_class = class_names[predicted.item()]
print(f'Predicted class: {predicted_class}')
