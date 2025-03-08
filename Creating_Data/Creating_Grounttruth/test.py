import os
import json
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder = r"D:\data_use\2024_11_22_08_52_03\camera_back"
output_folder = r"D:\data_use\2024_11_22_08_52_03\camerabackk_Obj_detection"

os.makedirs(output_folder, exist_ok=True)

model = YOLO("yolov8s.pt")

vehicle_class_ids = [1, 2, 3,4, 5, 6, 7, 8]
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

def process_image(image_path, image_name):
    results = model.predict(
        source=image_path,
        device=device,
        verbose=False
    )
    
    predictions = results[0].boxes
    objects_info = []
    for box in predictions:
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if cls_id in vehicle_class_ids:
            objects_info.append({
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, "unknown"),
                "bbox": [x1, y1, x2, y2]
            })

    json_filename = f"{image_name}_gt.json"
    output_path = os.path.join(output_folder, json_filename)
    with open(output_path, 'w') as f:
        json.dump(objects_info, f, indent=4)

if __name__ == "__main__":
    for image_name in os.listdir(data_folder):
        if image_name.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(data_folder, image_name)
            base_name = os.path.splitext(image_name)[0]
            process_image(image_path, base_name)
    
    print("Done")
