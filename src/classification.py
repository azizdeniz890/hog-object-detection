"""
Problem 3: Sınıflandırma ve Performans Analizi
Bu modül, HOG + SVM sınıflandırma sisteminin başarısını ölçer ve karşılaştırır.
Üretilen Metrikler: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

PDF Referansları:
- Sınıflandırma Sistemi: Bölüm 5 
- Rapor Grafikleri: Bölüm 6 
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog

# --- AYARLAR ---
BASE_DIR = "data/car_detection"
POS_DIR = os.path.join(BASE_DIR, "training_set", "positive")
NEG_DIR = os.path.join(BASE_DIR, "training_set", "negative")
REPORT_FIG_DIR = os.path.join("report", "figures")

# Grafiklerin kaydedileceği klasör yoksa oluştur
os.makedirs(REPORT_FIG_DIR, exist_ok=True)

def extract_hog_features(image):
    """Görüntüden HOG özelliklerini çıkarır (64x64 standardı)."""
    # HOG parametreleri Problem 1 ve 2 ile aynı olmalı
    image = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        transform_sqrt=True, 
        feature_vector=True
    )
    return features

def load_dataset():
    """Pozitif ve negatif klasörlerden tüm veriyi yükler."""
    print("[INFO] Veri seti yükleniyor...")
    data = []
    labels = []

    # 1. Pozitif (Arabalar) -> Etiket: 1
    if os.path.exists(POS_DIR):
        files = [f for f in os.listdir(POS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for filename in files:
            img = cv2.imread(os.path.join(POS_DIR, filename))
            if img is not None:
                data.append(extract_hog_features(img))
                labels.append(1)
    
    # 2. Negatif (Araba Olmayanlar) -> Etiket: 0
    if os.path.exists(NEG_DIR):
        files = [f for f in os.listdir(NEG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for filename in files:
            img = cv2.imread(os.path.join(NEG_DIR, filename))
            if img is not None:
                data.append(extract_hog_features(img))
                labels.append(0)

    print(f"[INFO] Toplam Veri Sayısı: {len(data)} (Pozitif: {labels.count(1)}, Negatif: {labels.count(0)})")
    return np.array(data), np.array(labels)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Confusion Matrix çizer ve kaydeder."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Diğer", "Araba"], 
                yticklabels=["Diğer", "Araba"])
    plt.title(title)
    plt.ylabel("Gerçek Etiket")
    plt.xlabel("Tahmin Edilen Etiket")
    
    save_path = os.path.join(REPORT_FIG_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[GRAFİK] Kaydedildi: {save_path}")

def compare_classifiers(X_train, X_test, y_train, y_test):
    """
    Problem 3 kapsamında farklı sınıflandırıcıları karşılaştırır.
    PDF: "Farklı yaklaşımları karşılaştıracaksınız." 
    """
    classifiers = {
        "SVM (Linear)": LinearSVC(random_state=42, max_iter=2000),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("\n--- SINIFLANDIRICI KARŞILAŞTIRMASI ---")
    
    for name, clf in classifiers.items():
        print(f"\n[EĞİTİM] {name} modeli eğitiliyor...")
        clf.fit(X_train, y_train)
        
        print(f"[TEST] {name} test ediliyor...")
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f">>> {name} Doğruluk (Accuracy): %{acc*100:.2f}")
        print(classification_report(y_test, y_pred, target_names=["Diğer", "Araba"]))
        
        # Sadece SVM için Confusion Matrix kaydedelim (Ana modelimiz)
        if name == "SVM (Linear)":
            plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {name}", "confusion_matrix_svm.png")

    # Karşılaştırma Grafiği
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = [v * 100 for v in results.values()]
    
    plt.bar(names, values, color=['blue', 'green', 'orange'])
    plt.ylim(0, 100)
    plt.ylabel("Doğruluk (%)")
    plt.title("Sınıflandırma Algoritmalarının Karşılaştırılması")
    
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"%{v:.1f}", ha='center')
        
    save_path = os.path.join(REPORT_FIG_DIR, "model_comparison.png")
    plt.savefig(save_path)
    print(f"[GRAFİK] Karşılaştırma grafiği kaydedildi: {save_path}")

if __name__ == "__main__":
    # 1. Veriyi Yükle
    X, y = load_dataset()
    
    if len(X) == 0:
        print("[HATA] Veri bulunamadı! Lütfen önce verileri indirin.")
    else:
        # 2. Veriyi Böl (%80 Eğitim, %20 Test)
        # stratify=y, her iki sette de araba/diğer oranının eşit olmasını sağlar
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 3. Modelleri Eğit ve Karşılaştır
        compare_classifiers(X_train, X_test, y_train, y_test)
        
        print("\n[BİLGİ] Problem 3 tamamlandı. Rapor için gerekli tüm grafikler 'report/figures' klasöründe.")