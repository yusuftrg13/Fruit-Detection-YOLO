from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

# Eğitilen modeli yükle
print("Model yükleniyor...")
model = YOLO('runs/detect/fruit_detection_v12/weights/best.pt')  # en iyi ağırlıklar

# Meyve sınıf isimleri
class_names = ["elma", "avokado", "muz", "guava", "kiwi", "mango", "portakal", "şeftali", "ananas"]

# Kamera başlatma
print("Kamera başlatılıyor...")
cap = cv2.VideoCapture(0)  # 0 numaralı kamera (varsayılan webcam)

# Kameranın açılıp açılmadığını kontrol et
if not cap.isOpened():
    print("HATA: Kamera açılamadı!")
    exit()

print("Kamera açıldı. Meyvelerinizi kameraya gösterin...")
print("Çıkmak için 'q' tuşuna basın.")

# FPS hesaplama için değişkenler
prev_time = 0
fps = 0

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()
    
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break
    
    # FPS hesapla
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) != 0 else 0
    prev_time = current_time
    
    # Model ile tahmin yap
    results = model(frame)
    
    # Tahminleri görselleştir
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    
    # Tespit edilen nesneler için çerçeve ve etiket çizme
    for box, cls_id, conf in zip(boxes, classes, confidences):
        # Eğer güven skoru 0.5'ten yüksekse çiz
        if conf > 0.4:  # 0.4 olarak değiştirildi, daha fazla tespit için
            x1, y1, x2, y2 = box
            class_name = class_names[cls_id]
            label = f"{class_name}: {conf:.2f}"
            
            # Sınıfa göre farklı renk ata
            color = [(42, 183, 202), (0, 215, 255), (50, 205, 50), 
                    (0, 255, 127), (255, 191, 0), (255, 20, 147), 
                    (0, 69, 255), (238, 130, 238), (128, 0, 128)][cls_id % 9]
            
            # Çerçeve çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Etiket arka planı
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
            
            # Etiketi çiz
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # FPS göster
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Kullanım talimatlarını göster
    cv2.putText(frame, "Cikmak icin 'q' tusuna basin", (10, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Görüntüyü göster
    cv2.imshow("Meyve Tespiti - Kamera", frame)
    
    # 'q' tuşuna basılırsa döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
print("Kamera kapatıldı.")
