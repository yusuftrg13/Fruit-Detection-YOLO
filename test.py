from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image

# Eğitilen modeli yükle
model = YOLO('runs/detect/fruit_detection_v12/weights/best.pt')  # en iyi ağırlıklar

# Meyve sınıf isimleri
class_names = ["elma","avokado","muz","guava","kiwi","mango","portakal","şeftali","ananas"]

# Test için bir görsel klasörü
test_folder = 'fruit-detection-dataset/fruit-detection-dataset/images/val'
save_folder = 'predictions'

# Kaydedilecek klasörü oluştur
os.makedirs(save_folder, exist_ok=True)

# Test klasöründeki tüm görseller için tahmin yap
for img_file in os.listdir(test_folder):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_folder, img_file)
        
        # Tahmin yap
        results = model(img_path)
        
        # Görüntüyü yükle
        img = cv2.imread(img_path)
        
        # Tahminleri görselleştir
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        # Her tespit edilen nesne için kutular çiz
        for box, cls_id, conf in zip(boxes, classes, confidences):
            # Eğer güven skoru 0.5'ten yüksekse çiz
            if conf > 0.5:
                x1, y1, x2, y2 = box
                class_name = class_names[cls_id]
                label = f"{class_name}: {conf:.2f}"
                
                # Kutuyu çiz
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Etiketi çiz
                cv2.putText(img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Tahmin sonucunu kaydet
        save_path = os.path.join(save_folder, f"pred_{img_file}")
        cv2.imwrite(save_path, img)

print(f"Tahminler '{save_folder}' klasöründe kaydedildi.")

# İnteraktif test için
while True:
    user_input = input("Test etmek için görüntü yolu girin (çıkmak için 'q'): ")
    
    if user_input.lower() == 'q':
        break
    
    if os.path.exists(user_input):
        results = model(user_input)
        
        # Sonuçları göster
        im_array = results[0].plot()  # BGR dizisine dönüştür
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL resmine dönüştür
        im.show()  # Görseli göster
    else:
        print("Dosya bulunamadı. Lütfen tekrar deneyin.")
