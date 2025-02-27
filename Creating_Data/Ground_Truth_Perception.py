import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np
import json
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_folder = "2024_11_18_b05"
output_folder = os.path.join(data_folder, "Final")
ROAD_CLASS_INDEX = 1

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "segmentation_masks"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "bounding_boxes"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "free_road_masks"), exist_ok=True)

segmentation_model = fcn_resnet50(pretrained=True).to(device).eval()
detection_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img

def save_segmentation_mask(mask, image_name):
    mask = mask[0].cpu().detach().numpy()
    mask = np.argmax(mask, axis=0)
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.save(os.path.join(output_folder, "segmentation_masks", f"{image_name}_segmentation.png"))

def save_bounding_boxes(bboxes, labels, image_name):
    bboxes = bboxes.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    result = []
    for bbox, label in zip(bboxes, labels):
        result.append({
            "bbox": bbox.tolist(),
            "label": label.item()
        })
    with open(os.path.join(output_folder, "bounding_boxes", f"{image_name}_bboxes.json"), 'w') as f:
        json.dump(result, f)

def save_road_mask(road_mask, image_name):
    road_mask = road_mask[0].cpu().detach().numpy()
    road_mask = (road_mask == ROAD_CLASS_INDEX).astype(np.uint8)
    road_mask_img = Image.fromarray(road_mask)
    road_mask_img.save(os.path.join(output_folder, "free_road_masks", f"{image_name}_road_mask.png"))

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_folder, image_name)
        image = load_image(image_path)

        with torch.no_grad():
            segmentation_output = segmentation_model(image)
        
        segmentation_mask = segmentation_output['out']
        
        with torch.no_grad():
            detection_output = detection_model(image)
        
        bboxes = detection_output[0]['boxes']
        labels = detection_output[0]['labels']

        if bboxes.size(0) == 0 or torch.sum(segmentation_mask) == 0:
            return None  # Loại bỏ ảnh không hợp lệ

        target = {
            'image': image,
            'segmentation_mask': segmentation_mask,
            'bboxes': bboxes,
            'labels': labels
        }

        return target

dataset = CustomDataset(data_folder)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: [d for d in x if d is not None])

segmentation_model.train()
detection_model.train()

optimizer = torch.optim.Adam([
    {'params': segmentation_model.parameters()},
    {'params': detection_model.parameters()}
], lr=1e-4)

num_epochs = 250
for epoch in range(num_epochs):
    for batch in dataloader:
        images = [item['image'] for item in batch]
        segmentation_masks = [item['segmentation_mask'] for item in batch]
        bboxes = [item['bboxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        optimizer.zero_grad()
        
        loss_segmentation = segmentation_model(images, labels=segmentation_masks)
        
        loss_detection = detection_model(images, targets=[{'boxes': b, 'labels': l} for b, l in zip(bboxes, labels)])

        loss = loss_segmentation + loss_detection['loss_classifier'] + loss_detection['loss_box_reg'] + loss_detection['loss_objectness']
        
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("Đã huấn luyện xong mô hình.")

