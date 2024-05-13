import torch
model = torch.load("./model/food_classification.pt")
from PIL import Image
from torchvision import transforms

class Label_encoder:
    def __init__(self, labels):
        labels = list(set(labels))
        self.labels = {label: idx for idx, label in enumerate(classes)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels[label]

def classify_image(image_path, model, label_encoder, device):
    # Load and preprocess the input image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)

    # Get predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()

    # Map index to class name
    predicted_label = label_encoder.get_label(predicted_idx)

    return predicted_label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = open("./dataset/classes.txt", 'r').read().splitlines()
label_encoder = Label_encoder(classes)

image_path = "dataset/images/test_set/kimbap/Img_069_0753.jpg"
predicted_label = classify_image(image_path, model, label_encoder, device)
print("Predicted Label:", predicted_label)