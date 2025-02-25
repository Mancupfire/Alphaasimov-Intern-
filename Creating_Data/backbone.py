import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class DetectionHead(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(64, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        batch_size, channels, h, w = x.shape
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, h, w)
        x = x.permute(0, 1, 3, 4, 2)
        return x

class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2, num_anchors=3):
        super(SimpleDetector, self).__init__()
        self.backbone = SimpleBackbone()
        self.head = DetectionHead(num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions

class SimpleDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted(os.listdir(img_dir))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        label_path = os.path.join(self.label_dir, self.img_files[index].replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, ndmin=2)
            labels = torch.from_numpy(labels)
        else:
            labels = torch.zeros((0, 5))
        return img, labels

    def __len__(self):
        return len(self.img_files)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), labels

def simple_loss(pred, targets):
    return torch.tensor(0.0)

def draw_boxes(img, preds, threshold=0.5):
    img = img.permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8).copy()
    preds = preds[0]
    for anchor in range(preds.shape[0]):
        for i in range(preds.shape[1]):
            for j in range(preds.shape[2]):
                box = preds[anchor, i, j]
                conf = box[4]
                if conf > threshold:
                    x, y, w, h = box[:4] * 640
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    dataset = SimpleDataset("data/images", "data/labels")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    model = SimpleDetector(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(5):
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            loss = simple_loss(preds, labels)
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            print(f"Predictions shape: {preds.shape}")
            break

    with torch.no_grad():
        img, _ = dataset[0]
        pred = model(img.unsqueeze(0).to(device))
        draw_boxes(img, pred.cpu())
