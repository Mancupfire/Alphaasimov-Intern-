import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_folder = "[Folder Here]"
output_folder = os.path.join(data_folder, "data_output")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "segmentation_masks"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "bounding_boxes"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "free_road_masks"), exist_ok=True)

segmentation_model = fcn_resnet50(pretrained=True).to(device).eval()
detection_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
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
    road_mask = (road_mask > 0.5).astype(np.uint8)  
    road_mask_img = Image.fromarray(road_mask)
    road_mask_img.save(os.path.join(output_folder, "free_road_masks", f"{image_name}_road_mask.png"))

def process_image(image_path, image_name):
    image = load_image(image_path)
    
    with torch.no_grad():
        output = segmentation_model(image)
        save_segmentation_mask(output['out'], image_name)
    
    with torch.no_grad():
        detection_output = detection_model(image)
        save_bounding_boxes(detection_output[0]['boxes'], detection_output[0]['labels'], image_name)

    road_mask = output['out'].cpu().detach().numpy()
    save_road_mask(road_mask, image_name)

for image_name in os.listdir(data_folder):
    if image_name.endswith(".jpg"): 
        image_path = os.path.join(data_folder, image_name)
        process_image(image_path, image_name)

print("Đã xử lý xong tất cả ảnh.")
