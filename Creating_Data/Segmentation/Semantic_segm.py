import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os


def unet_model(input_size=(480, 480, 3)):
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
    
    # Lớp output: sigmoid => xác suất cho lớp "có đường" (1)
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def preprocess_image(image_path, target_size=(480, 480)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc file ảnh: {image_path}")
    # Chuyển BGR -> RGB (tuỳ yêu cầu mô hình, thường RGB là chuẩn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize về target_size (480x480)
    image = cv2.resize(image, target_size)
    # Chuẩn hoá pixel về [0,1]
    image = image.astype('float32') / 255.0
    return image


def predict_segmentation(model, image_path, threshold=0.5):
    # Ảnh đầu vào có thể bất kỳ size => preprocess_image sẽ resize về 480x480
    image = preprocess_image(image_path, target_size=(480, 480))
    # Thêm chiều batch => (1, 480, 480, 3)
    image_input = np.expand_dims(image, axis=0)
    
    pred = model.predict(image_input)[0, ..., 0]  # shape (480, 480)
    
    # Chuyển sang mask nhị phân (0 hoặc 1) dựa trên threshold
    binary_mask = (pred > threshold).astype(np.uint8)
    return binary_mask


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
    # Khởi tạo U-Net với input_size 480x480
    model = unet_model(input_size=(480,480,3))
    # Compile mô hình
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # -------------------------------------------------
    #  (C) DỰ ĐOÁN CHO THƯ MỤC ẢNH ĐẦU VÀO
    # -------------------------------------------------
    # Folder chứa ảnh đầu vào
    input_folder = r"D:\data_use\2024_11_22_08_52_03\camera_front"
    
    # Folder lưu output
    output_folder = "Output_result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Ngưỡng (threshold) để chuyển xác suất -> nhị phân
    THRESHOLD = 0.5  # Bạn có thể thử 0.3 hoặc 0.2 nếu mask bị đen
    
    # Lấy danh sách file ảnh trong folder (lọc theo đuôi ảnh)
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    # Xử lý lần lượt các ảnh, đánh số thứ tự output
    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(input_folder, file_name)
        try:
            mask = predict_segmentation(model, image_path, threshold=THRESHOLD)
            # mask là mảng 0/1 cỡ (480,480)
            mask_to_save = (mask * 255).astype(np.uint8)
            # Lưu mask với tên mask_{idx}.png (hoặc bạn có thể tùy chỉnh)
            output_mask_path = os.path.join(output_folder, f"mask_{idx}.png")
            cv2.imwrite(output_mask_path, mask_to_save)
            print(f"Processed '{file_name}' -> '{output_mask_path}'")
        except Exception as e:
            print(f"Lỗi khi xử lý file {file_name}: {e}")
    
  
  
    # test_image_paths = [
    #     os.path.join(input_folder, 'anh_test1.jpg'),
    #     os.path.join(input_folder, 'anh_test2.jpg'),
    #     os.path.join(input_folder, 'anh_test3.jpg')
    # ]
    #
    # ground_truth_masks phải có cùng kích thước 480x480

    # ground_truth_masks = [np.zeros((480,480), dtype=np.uint8) for _ in test_image_paths]
    # mean_iou = evaluate_model(model, test_image_paths, ground_truth_masks, threshold=THRESHOLD)
    # print(f"Mean IoU trên tập kiểm tra: {mean_iou:.4f}")
