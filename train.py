from ultralytics import YOLO

# Model oluştur
model = YOLO('yolov8n.pt')  # yolov8nano, yolov8s, yolov8m, yolov8l, yolov8x modellerinden biri seçilebilir

# Modeli eğit
results = model.train(
    data='data.yaml',    epochs=10,  # Epoch sayısı azaltıldı (CPU eğitimi için)
    imgsz=416,   # Görüntü boyutu küçültüldü
    batch=4,    # Batch size azaltıldı (CPU için)
    device='cpu',    # GPU kullanmak için 0, CPU için 'cpu'
    name='fruit_detection_v1'  # Sonuçların kaydedileceği klasör adı
)

# Eğitim bittikten sonra modelin performansını değerlendirme
results = model.val()  # En son eğitilen modeli değerlendir
