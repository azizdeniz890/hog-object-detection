import os
import pickle
import numpy as np
import cv2
import urllib.request
import tarfile
import shutil

# CIFAR-10 indirme URL'si (Toronto Üniversitesi)
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
DATA_DIR = "data/cifar10_temp"

def download_and_extract():
    """CIFAR-10 veri setini indirir ve çıkarır."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    archive_path = os.path.join(DATA_DIR, ARCHIVE_NAME)
    
    if not os.path.exists(archive_path):
        print(f"[INFO] CIFAR-10 indiriliyor... ({URL})")
        urllib.request.urlretrieve(URL, archive_path)
        print("[INFO] İndirme tamamlandı.")
    
    print("[INFO] Dosyalar çıkarılıyor...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print("[INFO] Çıkarma tamamlandı.")

def unpickle(file):
    """CIFAR-10 batch dosyalarını okur."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images():
    """Resimleri positive ve negative klasörlerine kaydeder."""
    pos_dir = os.path.join("data", "training_set", "positive")
    neg_dir = os.path.join("data", "training_set", "negative")
    
    # Klasörleri temizle ve yeniden oluştur
    if os.path.exists(pos_dir): shutil.rmtree(pos_dir)
    if os.path.exists(neg_dir): shutil.rmtree(neg_dir)
    os.makedirs(pos_dir)
    os.makedirs(neg_dir)
    
    # CIFAR-10 meta verileri
    batch_path = os.path.join(DATA_DIR, "cifar-10-batches-py", "data_batch_1")
    data_dict = unpickle(batch_path)
    
    images = data_dict[b'data']
    labels = data_dict[b'labels']
    
    # CIFAR-10'da görüntüler (N, 3072) şeklindedir, (3, 32, 32)'ye çevrilmelidir.
    # Sonra (32, 32, 3) formatına (OpenCV formatı) dönüştürülür.
    images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    
    # Sınıf indeksi 1 = Automobile (Araba)
    CAR_CLASS = 1
    
    pos_count = 0
    neg_count = 0
    max_images = 100 # PDF en az 50 istiyor, biz 100 yapalım sağlam olsun
    
    print("[INFO] Görüntüler işleniyor ve kaydediliyor...")
    
    for i, img in enumerate(images):
        # OpenCV BGR formatını kullanır, CIFAR RGB'dir. Dönüştürelim.
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Görüntüyü biraz büyütelim (32x32 çok küçük, HOG için 64x64 yapalım)
        img_resized = cv2.resize(img_bgr, (64, 64))
        
        label = labels[i]
        
        if label == CAR_CLASS and pos_count < max_images:
            cv2.imwrite(os.path.join(pos_dir, f"pos_{pos_count}.png"), img_resized)
            pos_count += 1
            
        elif label != CAR_CLASS and neg_count < max_images:
            cv2.imwrite(os.path.join(neg_dir, f"neg_{neg_count}.png"), img_resized)
            neg_count += 1
            
        if pos_count >= max_images and neg_count >= max_images:
            break
            
    print(f"[SUCCESS] Tamamlandı!")
    print(f"Pozitif (Araba) Resim Sayısı: {pos_count} -> {pos_dir}")
    print(f"Negatif (Diğer) Resim Sayısı: {neg_count} -> {neg_dir}")
    
    # Geçici dosyaları temizle
    # shutil.rmtree(DATA_DIR) 

if __name__ == "__main__":
    download_and_extract()
    save_images()