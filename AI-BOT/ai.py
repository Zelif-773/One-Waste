from keras.models import load_model  # TensorFlow gerektirir
from PIL import Image, ImageOps  # Pillow kütüphanesi
import numpy as np
import os

# Bilimsel gösterimi kapat
np.set_printoptions(suppress=True)

# Modeli yükle
model = load_model("keras_model.h5", compile=False)

# Etiketleri (atık türlerini) yükle
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

# Görseli modelin beklediği formata hazırlamak için boş array oluştur
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Buraya atık resmi dosya yolunu yaz
image = Image.open("atik_resmi.jpg").convert("RGB")

# Görseli 224x224 boyutuna getir (merkezden kırparak)
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Görseli numpy array'e çevir
image_array = np.asarray(image)

# Normalize et (-1 ile 1 arasında değerler)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Model girişine hazır hale getir
data[0] = normalized_image_array

# Tahmin yap
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()  # Sınıf adı (atık türü)
confidence_score = prediction[0][index]

# Sonuçları yazdır
print(f"Atık Türü: {class_name}")
print(f"Güven Skoru: {confidence_score:.2f}")

# Aynı isimdeki .txt dosyasından bilgi oku (örnek: Organik.txt)
info_filename = f"{class_name}.txt"
if os.path.exists(info_filename):
    with open(info_filename, "r", encoding="utf-8") as f:
        info_text = f.read()
        print(f"Bilgi:\n{info_text}")
else:
    print("Bilgi dosyası bulunamadı.")
