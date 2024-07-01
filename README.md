# Dự Án Nhận Diện Đối Tượng Bằng YOLOv10

Đây là một dự án nhận diện đối tượng sử dụng YOLOv10 để phát hiện công nhân đội mũ bảo hiểm tại công trường xây dựng. Project này hướng dẫn bạn cách cài đặt và sử dụng model để phát hiện các đối tượng từ hình ảnh.

## Yêu Cầu Hệ Thống:

- Python 3.7 trở lên
- Google Colab (khuyến nghị để huấn luyện mô hình với GPU)
- Các thư viện cần thiết được liệt kê trong `requirements.txt`

## Hướng Dẫn Cài Đặt:

1. **Cài đặt Streamlit:**

   ```bash
   !pip install streamlit

2. **Clone repository YOLOv10:**

   ```bash
   !git clone https://github.com/THU-MIG/yolov10.git
   %cd yolov10


3. **Cài đặt các thư viện cần thiết**

   ```bash
    !pip install -q -r requirements.txt
    !pip install -e .

4. **Tải trọng số model YOLOv10**

   ```bash
    !wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt

5. **Tải và giải nén bộ dữ liệu**

   ```bash
    !gdown "https://drive.google.com/file/d/1twdtZEfcw4ghSZIiPDypJurZnNXzMO7R/view"
    !mkdir safety_helmet_dataset
    !unzip -q "/content/yolov10/Safety_Helmet_Dataset.zip" -d "/content/yolov10/safety_helmet_dataset"

## Huấn luyện mô hình:
    from ultralytics import YOLOv10
    YAML_PATH = "/content/yolov10/safety_helmet_dataset/Safety_Helmet_Dataset/  data.yaml"
    EPOCHS = 30
    IMG_SIZE = 100
    BATCH_SIZE = 16

    MODEL_PATH = "/content/yolov10/yolov10n.pt"
    model = YOLOv10(MODEL_PATH)
    model.train(data=YAML_PATH,
                epochs=EPOCHS,
                batch=BATCH_SIZE,
                imgsz=IMG_SIZE)

## Đánh giá mô hình: 
    from ultralytics import YOLOv10
    YAML_PATH = "/content/yolov10/safety_helmet_dataset/Safety_Helmet_Dataset/data.yaml"
    IMG_SIZE = 100

    TRAINED_MODEL_PATH = '/content/yolov10/runs/detect/train3/weights/best.pt'
    model = YOLOv10(TRAINED_MODEL_PATH)

    model.val(data=YAML_PATH,
            imgsz=IMG_SIZE,
            split='test')   

## Sử dụng mô hình để dự đoán: 
    from ultralytics import YOLOv10
    from google.colab.patches import cv2_imshow

    TRAINED_MODEL_PATH = '/content/yolov10/runs/detect/train3/weights/best.pt'
    model = YOLOv10(TRAINED_MODEL_PATH)

    IMAGE_URL = 'https://ips-dc.org/wp-content/uploads/2022/05/Black-Workers-Need-a-Bill-of-Rights.jpeg'
    CONF_THRESHOLD = 0.3
    results = model.predict(source=IMAGE_URL,
                            imgsz=IMG_SIZE,
                            conf=CONF_THRESHOLD,
                            max_det=100)

    annotated_img = results[0].plot()
    cv2_imshow(annotated_img)

## Lưu ý: 
- Chỉnh sửa đường dẫn và các tham số tùy thuộc vào cấu trúc thư mục và yêu cầu của bạn.
- Đảm bảo đã cài đặt tất cả các thư viện cần thiết trước khi chạy project.
