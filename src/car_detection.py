"""
Problem 2.2: Özel Nesne Tespiti (Sadece Araba)
Bu modül, HOG + SVM kullanarak araba tespiti yapar.
PDF Referansı: Bölüm 4.2 
"""

import cv2
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from skimage.feature import hog

# --- AYARLAR ---
# Bu dosyanın (car_detection.py) bulunduğu klasörü al
current_dir = os.path.dirname(os.path.abspath(__file__))
# Proje ana dizinine çık (src klasörünün bir üstü)
project_root = os.path.dirname(current_dir)

# Senin klasör yapına göre tam yolları oluştur
BASE_DIR = os.path.join(project_root, "data", "car_detection")
POS_DIR = os.path.join(BASE_DIR, "training_set", "positive")
NEG_DIR = os.path.join(BASE_DIR, "training_set", "negative")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(project_root, "models", "car_svm_model.pkl")

# Klasörleri oluştur (Garanti olsun)
os.makedirs(RESULTS_DIR, exist_ok=True)
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

# Yolları kontrol için yazdır
print(f"[BİLGİ] Eğitim Verisi Yolu: {POS_DIR}")

def non_max_suppression(boxes, scores, overlapThresh):
    """
    NMS: Çakışan kutuları güven skoruna (score) göre eler.
    """
    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Kutuları Y koordinatına göre değil, SKORLARINA göre sırala (Küçükten büyüğe)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        # En yüksek skora sahip kutuyu al (Listenin sonundaki)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Çakışma alanlarını bul
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Overlap oranını hesapla
        overlap = (w * h) / area[idxs[:last]]

        # Eşik değerinden fazla çakışanları sil
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), scores[pick]

def extract_hog_features(image):
    """Görüntüden HOG özelliklerini çıkarır."""
    image = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray, orientations=9, pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), block_norm='L2-Hys', 
        transform_sqrt=True, feature_vector=True
    )
    return features

def train_model():
    """Modeli eğitir."""
    print("[INFO] Eğitim verileri okunuyor...")
    data = []
    labels = []

    # Pozitif
    if not os.path.exists(POS_DIR):
        print(f"[HATA] Klasör yok: {POS_DIR}")
        return None
    for filename in os.listdir(POS_DIR):
        img_path = os.path.join(POS_DIR, filename)
        img = cv2.imread(img_path)
        if img is not None:
            data.append(extract_hog_features(img))
            labels.append(1)

    # Negatif
    for filename in os.listdir(NEG_DIR):
        img_path = os.path.join(NEG_DIR, filename)
        img = cv2.imread(img_path)
        if img is not None:
            data.append(extract_hog_features(img))
            labels.append(0)

    print(f"[INFO] Toplam {len(data)} resim ile model eğitiliyor (SVM)...")
    model = LinearSVC(random_state=42, max_iter=2000)
    model.fit(data, labels)
    
    if not os.path.exists("models"): os.makedirs("models")
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model kaydedildi: {MODEL_PATH}")
    return model

def sliding_window(image, step, window_size):
    h, w = image.shape[0], image.shape[1]
    y_max = h - window_size[1] + 1
    x_max = w - window_size[0] + 1
    if y_max <= 0 or x_max <= 0: return
    for y in range(0, y_max, step):
        for x in range(0, x_max, step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_cars(model, image_path):
    print(f"[INFO] Araba aranıyor: {image_path}")
    original_img = cv2.imread(image_path)
    if original_img is None: return

    # İşlem hızı için resmi çok büyükse baştan makul bir seviyeye çekelim
    base_img = original_img.copy()
    if base_img.shape[1] > 600:
        scale = 600 / base_img.shape[1]
        base_img = cv2.resize(base_img, (0, 0), fx=scale, fy=scale)

    # --- Multi-Scale (Piramit) Ayarları ---
    # Resmi farklı boyutlarda tarayacağız: %100, %80, %60 boyutlarında
    scales = [1.0, 0.8, 0.6] 
    
    window_size = (64, 64)
    step_size = 16 
    
    all_detections = []
    all_scores = []

    for scale in scales:
        # Resmi ölçekle
        resized_img = cv2.resize(base_img, (0, 0), fx=scale, fy=scale)
        
        # Eğer resim pencereden küçük kaldıysa bu ölçeği atla
        if resized_img.shape[0] < window_size[1] or resized_img.shape[1] < window_size[0]:
            continue

        # Bu ölçekte tarama yap
        for (x, y, window) in sliding_window(resized_img, step_size, window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
            
            features = extract_hog_features(window)
            pred = model.predict([features])[0]
            
            if pred == 1:
                score = model.decision_function([features])[0]
                if score > 0.5: # Eşik değeri
                    # Koordinatları orijinal resim boyutuna (base_img) geri çevirmeliyiz
                    # x_original = x / scale
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + window_size[0]) / scale)
                    y2 = int((y + window_size[1]) / scale)
                    
                    all_detections.append([x1, y1, x2, y2])
                    all_scores.append(score)

    # --- NMS UYGULAMA ADIMI ---
    detections_array = np.array(all_detections)
    scores_array = np.array(all_scores)
    
    # NMS uygula
    final_boxes, final_scores = non_max_suppression(detections_array, scores_array, overlapThresh=0.3)
    
    print(f"[DEBUG] Ham tespit: {len(all_detections)}, NMS sonrası: {len(final_boxes)}")

    # Sonuçları Çiz (Orijinal boyutlandırılmış base_img üzerine)
    # Not: Görselleştirme için base_img kullanıyoruz
    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(base_img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Kırmızı Kutu

    filename = os.path.basename(image_path)
    save_path = os.path.join(RESULTS_DIR, f"car_detected_{filename}")
    cv2.imwrite(save_path, base_img)
    print(f"[SONUÇ] Kaydedildi: {save_path}")

if __name__ == "__main__":
    # 1. Eğitim (Varsa yükler, yoksa eğitir - kodu basitleştirmek için her zaman eğitiyoruz şimdilik)
    model = train_model()
    
    # 2. Test
    if model:
        print(f"\n--- Test Başlıyor ---")
        files = [f for f in os.listdir(TEST_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
        for file in files:
            detect_cars(model, os.path.join(TEST_DIR, file))