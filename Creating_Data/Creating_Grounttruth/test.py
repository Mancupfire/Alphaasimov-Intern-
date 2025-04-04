import os
import json
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parent_folder = r"D:\data_use\2024_11_22_08_52_03"

camera_folders = [
    folder for folder in os.listdir(parent_folder)
    if os.path.isdir(os.path.join(parent_folder, folder)) and folder.startswith("camera")
]

model = YOLO("yolov8s.pt")

# Giá trị ngưỡng mặc định của YOLO (confidence threshold). Mặc định thường là 0.25.
default_confidence_threshold = 0.25
print(f"Default confidence threshold: {default_confidence_threshold}")

vehicle_class_ids = [1, 2, 3, 4, 5, 6, 7, 8]
class_names = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "human",
    5: "bus",
    6: "tree",
    7: "truck",
    8: "chair"
}

def process_image(image_path, image_name, output_folder):
    results = model.predict(
        source=image_path,
        device=device,
        verbose=False
    )
    
    predictions = results[0].boxes
    objects_info = []
    for box in predictions:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item() if hasattr(box, 'conf') and len(box.conf) > 0 else 1.0
        if conf >= default_confidence_threshold and cls_id in vehicle_class_ids:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects_info.append({
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, "unknown"),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
    
    json_filename = f"{image_name}_gt.json"
    output_path = os.path.join(output_folder, json_filename)
    with open(output_path, 'w') as f:
        json.dump(objects_info, f, indent=4)

if __name__ == "__main__":
    for camera_folder in camera_folders:
        data_folder = os.path.join(parent_folder, camera_folder)
        output_folder = os.path.join(parent_folder, f"{camera_folder}_Obj_detection")
        os.makedirs(output_folder, exist_ok=True)
        
        for image_name in os.listdir(data_folder):
            if image_name.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(data_folder, image_name)
                base_name = os.path.splitext(image_name)[0]
                process_image(image_path, base_name, output_folder)
        
        print(f"Xử lý xong folder: {camera_folder}")
    
    print("Tất cả các folder đã được xử lý xong!")
