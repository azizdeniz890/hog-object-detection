"""
Problem 2.1: İnsan Tespiti (Human Detection) - Final Fix
Bu modül, OpenCV'nin hazır HOG modelini kullanarak insan tespiti yapar.
Düzeltmeler: IndexError giderildi, küçük resim desteği eklendi.
"""

import cv2
import numpy as np
import os

# --- AYARLAR (Dinamik Yol Bulma) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
BASE_DIR = os.path.join(project_root, "data", "human_detection")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Klasörleri oluştur
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, overlapThresh: float) -> tuple:
    """Çakışan kutuları güven skoruna göre eler."""
    if len(boxes) == 0: return [], []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Skorlara göre sırala
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), scores[pick]

def detect_people(image_path: str, hit_threshold: float) -> int:
    if not os.path.exists(image_path): return 0
    image = cv2.imread(image_path)
    if image is None: return 0
        
    # --- RESİM BOYUTU AYARI (Kritik Düzeltme) ---
    # Eğer resim çok küçükse (INRIA crop resimleri gibi), HOG çalışmaz.
    # Bu yüzden küçük resimleri biraz büyütüyoruz.
    if image.shape[1] < 400: 
        scale_up = 2.0
        image = cv2.resize(image, (0, 0), fx=scale_up, fy=scale_up)
    elif image.shape[1] > 800:
        scale_down = 800 / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=scale_down, fy=scale_down)
    
    # HOG Başlat
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Tespit Yap
    try:
        (rects, weights) = hog.detectMultiScale(
            image, 
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05, 
            hitThreshold=hit_threshold
        )
    except Exception as e:
        print(f"[HATA] HOG işlenirken hata: {e}")
        return 0

    # --- HATA DÜZELTMESİ BURADA ---
    # Eğer hiç tespit yoksa hemen dön (IndexError'ı engeller)
    if len(rects) == 0:
        return 0

    # Verileri düzenle
    rects_array = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    
    # Flatten kullanarak boyut hatasını (IndexError) kesin çözeriz
    scores_array = np.array(weights).flatten()
    
    # NMS Uygula
    final_boxes, final_scores = non_max_suppression(rects_array, scores_array, overlapThresh=0.65)
    
    # Çizim
    for (xA, yA, xB, yB), score in zip(final_boxes, final_scores):
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        label = f"Human: {score:.2f}"
        cv2.putText(image, label, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Kaydet
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_thresh_{hit_threshold}{ext}"
    cv2.imwrite(os.path.join(RESULTS_DIR, output_filename), image)
    
    return len(final_boxes)

if __name__ == "__main__":
    print(f"[BİLGİ] Test Klasörü: {TEST_DIR}")
    print("-" * 50)
    
    files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not files:
        print("[UYARI] Test klasörü boş! Lütfen resim ekleyin.")
    else:
        thresholds = [0.1, 0.3, 0.6] 
        print(f"{'Dosya Adı':<25} | {'Threshold':<10} | {'Tespit Sayısı':<10}")
        print("-" * 55)
        
        for filename in files:
            in_path = os.path.join(TEST_DIR, filename)
            for thresh in thresholds:
                count = detect_people(in_path, thresh)
                print(f"{filename:<25} | {thresh:<10} | {count:<10}")
        
        print("-" * 55)
        print(f"[BAŞARILI] Sonuçlar '{RESULTS_DIR}' klasörüne kaydedildi.")