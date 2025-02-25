import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision import models
from PIL import Image

class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x


class DetectionHead(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.fc1 = nn.Linear(input_channels * 16 * 16, 1024)  
        self.fc2 = nn.Linear(1024, num_classes)  
        self.fc3 = nn.Linear(1024, 4) 

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        class_preds = self.fc2(x)
        bbox_preds = self.fc3(x)
        return class_preds, bbox_preds


class SimpleObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleObjectDetectionModel, self).__init__()
        self.backbone = SimpleBackbone()
        self.detection_head = DetectionHead(input_channels=256, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        class_preds, bbox_preds = self.detection_head(features)
        return class_preds, bbox_preds


num_classes = 20  
model = SimpleObjectDetectionModel(num_classes)

transform = transforms.Compose([transforms.ToTensor()])

dataset = VOCDetection(root='[Data Here]', year='[None yet]', image_set='train', download=True, transform=transform)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, targets in data_loader:
    model.eval()
    with torch.no_grad():
        class_preds, bbox_preds = model(images)

    print(f"Class Predictions: {class_preds}")
    print(f"Bounding Box Predictions: {bbox_preds}")

    break

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (bx2 - bx1) * (by2 - by1)
    
    inter_x1 = max(x1, bx1)
    inter_y1 = max(y1, by1)
    inter_x2 = min(x2, bx2)
    inter_y2 = min(y2, by2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area

for images, targets in data_loader:
    model.eval()
    with torch.no_grad():
        class_preds, bbox_preds = model(images)

    for i in range(len(bbox_preds)):
        iou = compute_iou(bbox_preds[i], targets[i]['annotation']['bndbox'])
        print(f"Mean IoU for Image {i}: {iou}")
    
      break
