# HOG (Histogram of Oriented Gradients) ile Nesne Tespiti ve Sınıflandırma
---

## 📋 Proje Özeti
Bu proje, HOG algoritmasının teorik temellerini anlamak, sıfırdan implementasyonunu gerçekleştirmek ve bu özellikleri kullanarak hem insan hem de araç tespiti yapabilen sistemler geliştirmek amacıyla hazırlanmıştır. Proje üç ana problemden oluşmaktadır:
1.  **HOG İmplementasyonu:** Algoritmanın (Gradyan, Histogram, Normalizasyon) manuel kodlanması.
2.  **Nesne Tespiti:** İnsan (OpenCV) ve Araç (Özel SVM Eğitimi) tespiti.
3.  **Sınıflandırma Analizi:** Farklı makine öğrenmesi algoritmalarının performans karşılaştırması.

---

## 📂 Dosya ve Klasör Yapısı

```text
aziz-deniz-akmermer-hog-odev/
│
├── README.md               <-- (Şu an okuduğunuz dosya)
├── requirements.txt        <-- Gerekli Python kütüphaneleri
│
├── src/                    <-- Kaynak Kodlar
│   ├── hog_implementation.py   # Problem 1: HOG algoritması ve görselleştirme
│   ├── human_detection.py      # Problem 2.1: İnsan tespiti (OpenCV Model)
│   ├── car_detection.py        # Problem 2.2: Araç tespiti (Custom SVM Model)
│   └── classification.py       # Problem 3: Performans analizi ve grafikler
│
├── data/                   <-- Veri Setleri ve Sonuçlar
│   ├── human_detection/
│   │   ├── test_images/        # İnsan tespiti test görselleri
│   │   └── results/            # İnsan tespiti sonuçları (Yeşil kutular)
│   │
│   └── car_detection/
│       ├── training_set/       # Eğitim verileri (500+ Pozitif, 500+ Negatif)
│       ├── test_images/        # Araç tespiti test görselleri
│       └── results/            # Araç tespiti sonuçları (Kırmızı kutular)
│
├── models/                 <-- Eğitilen Modeller
│   └── car_svm_model.pkl       # Eğitilmiş araç tespit modeli
│
# HOG (Histogram of Oriented Gradients) ile Nesne Tespiti ve Sınıflandırma

**Ders:** Bilgisayarla Görü

**Öğrenci Adı:** Aziz Deniz Akmermer

**Öğrenci No:** 220212037

**Bölüm:** Yapay Zeka Mühendisliği

---

## Proje Özeti
Bu proje HOG algoritmasının teorik temellerini anlamayı, algoritmayı sıfırdan uygulamayı ve elde edilen
HOG özelliklerini kullanarak hem insan hem de araç tespiti gerçekleştiren sistemler geliştirmeyi amaçlar.
Proje üç temel bölümden oluşur:

1. HOG uygulaması: Gradyan, histogram ve normalizasyon adımlarının manuel implemantasyonu ve görselleştirmesi.
2. Nesne tespiti: İnsan tespiti (OpenCV hazır model) ve araç tespiti (HOG ile çıkarılan özellikler ile LinearSVC).
3. Sınıflandırma analizi: Farklı sınıflandırıcıların (SVM, k‑NN, Random Forest) performans karşılaştırması.

Detaylı deneyler ve sonuçlar `report/report.pdf` içinde yer almaktadır.

---

## Dosya ve Klasör Yapısı

```
aziz-deniz-akmermer-hog-odev/
│
├── README.md
├── requirements.txt
├── src/
│   ├── hog_implementation.py
│   ├── human_detection.py
│   ├── car_detection.py
│   └── classification.py
├── data/
│   ├── human_detection/
│   │   ├── test_images/
│   │   └── results/
│   └── car_detection/
│       ├── training_set/
│       │   ├── positive/
│       │   └── negative/
│       ├── test_images/
│       └── results/
├── models/
│   └── car_svm_model.pkl
└── report/
        ├── report.pdf
        └── figures/
```

Not: Repo içinde alternatif düzenler (`data/car_detection/...`) bulunabilir; scriptler bu alternatifleri
kontrol edecek şekilde yazılmıştır. İsterseniz veriyi README'de belirtilen standart yapıya taşıyabilirsiniz.

---

## Kurulum

Aşağıdaki adımlar Windows PowerShell için örnektir. Paketleri projenin çalıştırılacağı Python ortamına kurduğunuzdan emin olun.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Temel paketler: `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

Eğer birden fazla Python kurulumu varsa, `python --version` ve `where.exe python` ile kullanılan yorumlayıcıyı doğrulayın.

---

## Kullanım

Proje dizininden aşağıdaki komutlarla her problem için ayrı çalıştırma yapabilirsiniz.

### 1) HOG İmplementasyonu (Problem 1)

```powershell
python src/hog_implementation.py
```

çıktı: Görselleştirme ve örnek HOG görselleri proje dizinine kaydedilebilir.

### 2) İnsan Tespiti (Problem 2.1)

```powershell
python src/human_detection.py
```

Girdi: `data/human_detection/test_images/` içindeki görseller
Çıktı: `data/human_detection/results/` içine kaydedilen tespit görselleri.

### 3) Araç Tespiti (Problem 2.2)

```powershell
python src/car_detection.py
```

İşleyiş: Script önce `data/car_detection/training_set/positive` ve `negative` içindeki verilerle modeli eğitir,
ardından `data/car_detection/test_images/` içindeki görsellerde sliding-window ile tespit gerçekleştirir ve
sonuçları `data/car_detection/results/` klasörüne kaydeder.

### 4) Sınıflandırma ve Analiz (Problem 3)

```powershell
python src/classification.py
```

Çıktı: Terminalde sınıflandırıcı karşılaştırmaları; `report/figures/` içine kaydedilen grafikler.

---

## Veri Hazırlama ve Öneriler

- Pozitif/negatif örneklerin dengeli olmasına dikkat edin (öneri: en az 50+50; örnekte 500/500 kullanılmıştır).
- Pozitif örnekler farklı ölçek, açı ve arka plan çeşitliliği içermelidir.
- Veri artırma (augmentation) ile model genelleştirmesini iyileştirebilirsiniz (ölçekleme, döndürme, parlaklık vb.).

---

## Teknik Detaylar

- HOG özellikleri: `skimage.feature.hog` veya proje içi manuel HOG implementasyonu.
- Sınıflandırıcı: `sklearn.svm.LinearSVC` ile eğitim yapılır; model `joblib` ile kaydedilir.
- Sliding-window: Sabit pencere boyutu (64×64) ile tarama; kod küçük görüntüler için upsample, büyük görüntüler için
    isteğe bağlı downsample yapar.
- Post-processing: Non-Maximum Suppression (NMS) ile çakışan kutular birleştirilebilir.

---

## Sorunlar ve Çözümler

- `ModuleNotFoundError`: Paketleri doğru Python ortamına kurduğunuzdan emin olun (`python -m pip install -r requirements.txt`).
- `0 pencere tarandı` veya test atlanıyor: Sliding-window pencere boyutu ile görüntü boyutu uyuşmuyordur. Kod upsampling
    uygulasa da alternatif olarak `window_size` ve `step_size` parametrelerini küçültmeyi deneyin.
- Düşük tespit oranı: Eğitim verisini çeşitlendirin, multi-scale detection (piramid) ve hard-negative mining uygulayın.

---

## Deneysel Sonuçlar (Özet)

- Kullanılan veri seti örneği: 1000 görüntü (500 pozitif, 500 negatif).
- Araç tespiti için örnek doğruluk: ~%98.9 (detaylar raporda).

Detaylı metrikler ve grafikler `report/report.pdf` ve `report/figures/` içindedir.

---

## İleri Çalışmalar ve Öneriler

- Multi-scale detection: Görüntü piramidi ile farklı ölçeklerde tarama; sonuçları NMS ile birleştirme.
- HOG parametreleri ile deneyler (pixels_per_cell, cells_per_block, orientations).
- Derin öğrenme tabanlı özellikler veya transfer öğrenme ile performansı arttırma.

---

## İletişim

Aziz Deniz Akmermer — 220212037

---
