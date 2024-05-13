from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.detection import maskrcnn_resnet50_fpn

image_folder_path = 'dataset/images/train_set/'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = ImageFolder(root=image_folder_path, transform=transform)

class_to_idx = dataset.class_to_idx
print("Classes and their indices:", class_to_idx)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

class IMAGE_CNN(nn.Module):
    def __init__(self):
        super(IMAGE_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(524288, 128)
        self.fc2 = nn.Linear(128, 14)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # x = self.relu(self.conv2(x))
        # x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model = IMAGE_CNN()

model = maskrcnn_resnet50_fpn()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)

num_epochs = 5

# print(len(dataloader))

loop = 0

#training loop
for epoch in range(num_epochs):
    loop = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        #images = list(image.to(device) for image in images)
        #labels = [t for t in labels]
        boxes = torch.zeros([len(labels),4], dtype=torch.float)
        masks = torch.zeros([10,256,256], dtype=torch.float32)
        print(labels)
        rcnn_labels = []
        print(images.shape)
        
        # rcnn_labels["boxes"] = boxes
        # rcnn_labels["labels"] = torch.ones((10,), dtype=torch.int64) 
        # rcnn_labels["masks"] = masks
        rcnn_labels.append({"boxes":boxes})
        rcnn_labels.append({"labels":torch.ones((10,), dtype=torch.int64)})
        rcnn_labels.append({"masks":masks})
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        #print(rcnn_labels)
        
        outputs = model(images, rcnn_labels)
        # print(outputs)
        loss = sum(loss for loss in outputs.values())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        print("loop number %d" % (loop))
        loop += 1
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')