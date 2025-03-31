import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os


def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder (Phần mã hóa)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)
    
    # Bridge (Phần nối giữa encoder và decoder)
    c5 = Conv2D(1024, (3,3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3,3), activation='relu', padding='same')(c5)
    
    # Decoder (Phần giải mã)
    u6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3,3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3,3), activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3,3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3,3), activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3,3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3,3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc file ảnh: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    return image


def predict_segmentation(model, image_path, threshold=0.5):
    image = preprocess_image(image_path, target_size=(256, 256))
    image_input = np.expand_dims(image, axis=0)
    
    pred = model.predict(image_input)[0, ..., 0]  # shape (256, 256)
    
    binary_mask = (pred > threshold).astype(np.uint8)
    binary_map = binary_mask * 255
    return binary_map


def evaluate_model(model, image_paths, ground_truth_masks, threshold=0.5):
    iou_scores = []
    for image_path, gt_mask in zip(image_paths, ground_truth_masks):
        pred_mask = predict_segmentation(model, image_path, threshold=threshold)
        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 1.0
        iou_scores.append(iou)
    mean_iou = np.mean(iou_scores)
    return mean_iou


def train_model(model, X_train, Y_train, batch_size=2, epochs=10):
    model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1
    )
    return model


if __name__ == '__main__':
    model = unet_model(input_size=(256, 256, 3))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    input_folder = r"D:\data_use\2024_11_22_08_52_03\camera_front"

    # Output folder to save results
    output_folder = "Output_results_here"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Threshold to convert probabilities to binary
    THRESHOLD = 0.5  # Try 0.3 or 0.2 if the mask is too dark

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(input_folder, file_name)
        try:
            # Predict segmentation mask with the given threshold
            mask = predict_segmentation(model, image_path, threshold=THRESHOLD)
            output_mask_path = os.path.join(output_folder, f"mask_{idx}.png")
            cv2.imwrite(output_mask_path, mask)
            print(f"Processed '{file_name}' -> '{output_mask_path}'")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

