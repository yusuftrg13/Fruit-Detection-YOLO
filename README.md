# YOLOv8 Meyve Tespiti Projesi

## Genel Bakış

Bu proje, YOLOv8 kullanarak 9 farklı meyve türünü (elma, avokado, muz, guava, kiwi, mango, portakal, şeftali ve ananas) tespit etmek için eğitilmiş bir yapay zeka modelini içermektedir.

## Model Performansı

- **Model**: YOLOv8n (nano)
- **Dataset**: Fruit Detection Dataset
- **Sınıflar**: 9 meyve türü
- **Performans Metrikleri**:
  - mAP50: %67.18
  - Precision: %64.28
  - Recall: %61.46
  - mAP50-95: %46.88

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install ultralytics opencv-python matplotlib pillow
```

2. Bu repo'yu klonlayın:
```bash
git clone https://github.com/yusuftrg13/FRU-T-DETECT-ON.git
cd FRU-T-DETECT-ON
```

## Kullanım

### Eğitim

Modeli yeniden eğitmek için:
```bash
python train.py
```

### Test

Doğrulama setindeki görselleri test etmek için:
```bash
python test.py
```

### Kamera ile Canlı Test

Bilgisayar kameranız ile canlı meyve tespiti için:
```bash
python kamera_test.py
```

## Veri Seti Yapısı

Veri seti, iki ana dizinde düzenlenmiştir:

* **`images/`**: Bu dizin, çeşitli formatlarda (örn. `.jpg`, `.jpeg`, `.png`) tüm görüntü dosyalarını içerir. Görüntüler farklı boyutlardadır ve çeşitli ortamlarda ve ışık koşullarında tek veya çoklu meyve örneklerini gösterir. Bu dizin şu alt dizinlere ayrılmıştır:
    * **`train/`**: Model eğitimi için kullanılan görüntüleri içerir (toplam görüntülerin yaklaşık %80'i).
    * **`val/`**: Eğitim sırasında modeli doğrulamak için kullanılan görüntüleri içerir (toplam görüntülerin yaklaşık %20'si).

* **`labels/`**: Bu dizin, `images/` dizinindeki görüntülere karşılık gelen etiket dosyalarını içerir. Etiket dosyaları, görüntülerdeki her bir meyvenin konumu ve sınıfı hakkında gerçek bilgileri sağlar. Etiketleme formatı **YOLO formatındadır** (görüntü dosyalarıyla aynı isimde `.txt` uzantılı dosyalar). Bu dizin de şu alt dizinlere ayrılmıştır:
    * **`train/`**: Eğitim görüntüleri için etiket dosyalarını içerir.
    * **`val/`**: Doğrulama görüntüleri için etiket dosyalarını içerir.

Örneğin, `images/train/apple_1.jpg` görüntüsü için etiket dosyası `labels/train/apple_1.txt` olacaktır.

## Etiketleme Formatı (YOLO)

`labels/` dizinindeki her bir etiket dosyası (`.txt`), `images/` dizinindeki aynı isimli bir görüntüye karşılık gelir ve o görüntüdeki her bir tespit edilen meyve örneği için bir satır içerir. Her satırın formatı aşağıdaki gibidir:

``` <sınıf_id> <x_merkez> <y_merkez> <genişlik> <yükseklik> ```

Burada:

* **`<sınıf_id>`**: Meyvenin sınıfını temsil eden bir tam sayı. Bu veri setindeki sınıf ID'leri ve meyve adları eşleştirmesi şöyledir:
    * `0`: elma
    * `1`: avokado
    * `2`: muz
    * `3`: guava
    * `4`: kiwi
    * `5`: mango
    * `6`: portakal
    * `7`: şeftali
    * `8`: ananas
* **`<x_merkez>`**: Sınırlayıcı kutunun merkezinin normalleştirilmiş x-koordinatı (0 ile 1 arasında).
* **`<y_merkez>`**: Sınırlayıcı kutunun merkezinin normalleştirilmiş y-koordinatı (0 ile 1 arasında).
* **`<genişlik>`**: Sınırlayıcı kutunun normalleştirilmiş genişliği (0 ile 1 arasında).
* **`<yükseklik>`**: Sınırlayıcı kutunun normalleştirilmiş yüksekliği (0 ile 1 arasında).

**Not:** Tüm koordinatlar ve boyutlar, sırasıyla görüntünün genişliği ve yüksekliğine göre normalleştirilmiştir. Bu etiketlemeler Label Studio kullanılarak oluşturulmuştur.

## Veri Seti İstatistikleri

* **Sınıf Sayısı:** 9 (elma, avokado, muz, guava, kiwi, mango, portakal, şeftali, ananas)
* **Toplam Görüntü Sayısı:** Yaklaşık 800+ görüntü
    * Eğitim Seti: Toplam görüntülerin yaklaşık %80'i
    * Doğrulama Seti: Toplam görüntülerin yaklaşık %20'si
* **Görüntü Boyutları:** Görüntüler çeşitli orijinal boyutlardadır.
* **Sınıf Dağılımı:** Veri seti, farklı meyve sınıflarının çeşitli dağılımına sahiptir, her sınıf için dengeli bir şekilde örnekler içermektedir.

## Veri Ön İşleme ve Zenginleştirme

**Mevcut Durum:** Bu veri setindeki görüntülere açıkça herhangi bir ön işleme veya veri zenginleştirme uygulanmamıştır.

**Gelecek Adımlar:** Modelin performansını ve genelleştirme yeteneğini iyileştirmek için eğitim sürecinde uygun ön işleme adımları (örn. model için tutarlı bir giriş boyutuna yeniden boyutlandırma) ve veri zenginleştirme teknikleri (örn. rastgele kırpma, çevirme, döndürme, renk ayarları) uygulanması önerilir.

## Kullanım Amacı

Bu veri seti, özellikle belirtilen dokuz meyve sınıfını tanımlamak için nesne tespit modellerini eğitmek ve değerlendirmek amacıyla hazırlanmıştır. Şu gibi görevler için kullanılabilir:

* Görüntü ve videolarda gerçek zamanlı meyve tespiti.
* Çeşitli uygulamalarda otomatik meyve tanıma.
* Bu belirli meyve seti üzerinde nesne tespit modellerinin performansını karşılaştırma.

## Model Oluşturma Süreci

1. YOLOv8n temel modelini kullanarak projeyi başlattık
2. Veri setini doğru şekilde yapılandırdık
3. Model eğitimini CPU üzerinde 10 epoch ile gerçekleştirdik
4. Test ve doğrulama adımlarını tamamladık
5. Kamera entegrasyonu ile gerçek zamanlı tespit özelliği ekledik

## Teşekkürler

Etiketleme için Label Studio kullanılmıştır.

## İletişim

GitHub: [yusuftrg13](https://github.com/yusuftrg13)

## Model İndirme Linki

Eğitilmiş model ağırlıkları (YOLOv8n) 100MB'dan büyük olduğu için GitHub'a yüklenememiştir. Modeli aşağıdaki linkten indirebilirsiniz:

[YOLOv8 Meyve Tespiti Model Ağırlıkları - Google Drive](https://drive.google.com/BURAYA_GOOGLE_DRIVE_LINKINIZI_EKLEYIN)

İndirdikten sonra, modeli `runs/detect/fruit_detection_v12/weights/best.pt` konumuna yerleştirin.