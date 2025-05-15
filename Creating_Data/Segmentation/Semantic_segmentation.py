import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO

COLOR_MAP = {
    0:  (0, 0, 0),    1:  (128, 64,128), 2:  (244, 35,232), 3:  (70, 70, 70),
    4:  (102,102,156), 5:  (190,153,153), 6:  (153,153,153), 7:  (250,170,30),
    8:  (220,220,0),  9:  (107,142,35), 10: (152,251,152), 11: (70,130,180),
   12: (220,20,60), 13: (255,0,0),      14: (0,0,142),      15: (0,0,70),
   16: (0,60,100),  17: (0,80,100),    18: (0,0,230),     19: (119,11,32)
}

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=len(COLOR_MAP)):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool(c1)
        c2 = self.enc2(p1)
        p2 = self.pool(c2)
        c3 = self.enc3(p2)
        p3 = self.pool(c3)
        c4 = self.enc4(p3)
        p4 = self.pool(c4)
        bn = self.bottleneck(p4)
        u4 = self.upconv4(bn)
        d4 = self.dec4(torch.cat([u4, c4], dim=1))
        u3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))
        return self.conv_last(d1)


def preprocess_image_torch(path, device, target_size=(480,480)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    tensor = torch.from_numpy(img).float().permute(2,0,1) / 255.0
    return tensor.unsqueeze(0).to(device)


def predict_mask_torch(model, img_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits)
            mask = (prob>0.5).cpu().squeeze(0).squeeze(0).numpy().astype(np.uint8)
        else:
            mask = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().squeeze(0).numpy().astype(np.uint8)
    return mask


def save_color_mask(mask, save_path):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        color_mask[mask == cls_id] = color
    cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

def get_camera_folders(data_dir, prefix, exclude_substr=None):
    folders = []
    for root, subs, _ in os.walk(data_dir):
        for sub in subs:
            if sub.startswith(prefix) and (exclude_substr is None or exclude_substr not in sub):
                folders.append(os.path.join(root, sub))
    base = os.path.basename(data_dir.rstrip(os.sep))
    if not folders and base.startswith(prefix):
        folders.append(data_dir)
    return folders

def process_image_yolo(model, img_path, base, out_folder, device, conf, filter_ids=None):
    results = model.predict(source=img_path, device=device, verbose=False, conf=conf)
    dets = []
    for box in results[0].boxes:
        cid = int(box.cls[0]); c = float(box.conf[0])
        if filter_ids and cid not in filter_ids:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        dets.append({'class_id': cid, 'confidence': c, 'bbox': [x1, y1, x2, y2]})
    if dets:
        path = os.path.join(out_folder, f"{base}_gt.json")
        with open(path, 'w') as f:
            json.dump(dets, f, indent=4)
    return dets

def display_table(out_folder):
    records = []
    for f in os.listdir(out_folder):
        if f.endswith('_gt.json'):
            img = f.replace('_gt.json','')
            data = json.load(open(os.path.join(out_folder,f)))
            for o in data:
                records.append({'image': img, **o})
    if records:
        print(pd.DataFrame(records).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Segmentation+Detection pipeline')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--yolo_weights', '-y', default='yolo12x.pt')
    parser.add_argument('--prefix', default='camera')
    parser.add_argument('--exclude', default=None)
    parser.add_argument('--out_suf', default='od_gt')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--remove_no', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--seg_weights', default=None, help='UNet .pth weights')
    parser.add_argument('--seg', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(args.yolo_weights, device=device)
    if args.seg:
        seg = UNet().to(device)
        if args.seg_weights:
            seg.load_state_dict(torch.load(args.seg_weights, map_location=device))

    cams = get_camera_folders(args.data_dir, args.prefix, args.exclude)
    for cam in cams:
        out_folder = os.path.join(os.path.dirname(cam), os.path.basename(cam) + '_' + args.out_suf)
        os.makedirs(out_folder, exist_ok=True)
        imgs = [f for f in os.listdir(cam) if f.lower().endswith(('.jpg','.png'))]
        for img in imgs:
            path = os.path.join(cam, img)
            base = os.path.splitext(img)[0]
            if args.seg:
                tensor = preprocess_image_torch(path, device)
                mask = predict_mask_torch(seg, tensor)
                save_color_mask(mask, os.path.join(out_folder, base + '_seg.png'))
            dets = process_image_yolo(
                yolo, path, base, out_folder, device, args.conf,
                filter_ids=[1,2,3,5,6,7,8] if args.filter else None
            )
            if args.remove_no and not dets:
                os.remove(path)
        if args.show:
            display_table(out_folder)

if __name__ == '__main__':
    main()
