"""
Problem 1: HOG Implementasyonu
Bu modül, HOG (Histogram of Oriented Gradients) algoritmasının temel adımlarını
sıfırdan implement eder.
"""

import numpy as np
import cv2

def compute_gradients(image: np.ndarray) -> tuple:
    """
    Görüntünün x ve y yönündeki gradyanlarını, büyüklüğünü ve yönelimini hesaplar.
    
    Args:
        image (np.ndarray): Gri tonlamalı girdi görüntüsü (H, W).
        
    Returns:
        tuple: (magnitude, angle)
            - magnitude (np.ndarray): Gradyan büyüklüğü.
            - angle (np.ndarray): Gradyan açısı (derece cinsinden, 0-180 aralığında).
    """
    # Islem hassasiyeti icin float32'ye donusturme
    img_float = image.astype(np.float32)
    
    # 1. Gradyan Hesaplama
    kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)
    
    gx = cv2.filter2D(img_float, -1, kernel_x)
    gy = cv2.filter2D(img_float, -1, kernel_y)
    
    magnitude = np.sqrt(cv2.add(np.square(gx), np.square(gy)))
    angle_rad = np.arctan2(gy, gx)
    angle_deg = np.degrees(angle_rad)
    angle_deg = angle_deg % 180
    
    return magnitude, angle_deg

def create_cell_histogram(cell_magnitude: np.ndarray, cell_angle: np.ndarray, num_bins: int = 9) -> np.ndarray:
    """
    Bir hücre için yönelim histogramı oluşturur.
    
    Args:
        cell_magnitude (np.ndarray): Hücre içindeki piksellerin gradyan büyüklükleri.
        cell_angle (np.ndarray): Hücre içindeki piksellerin gradyan açıları (0-180 derece).
        num_bins (int): Histogramdaki kutu (bin) sayısı. Varsayılan 9.
        
    Returns:
        np.ndarray: boyutunda histogram dizisi.
    """
    histogram = np.zeros(num_bins, dtype=np.float32)
    bin_width = 180.0 / num_bins
    
    # Her pikselin hangi bin'lerin arasinda oldugunu bul
    bin_indices = cell_angle / bin_width
    
    # Alt ve ust bin indeksleri
    bin_idx_1 = np.floor(bin_indices).astype(int)
    bin_idx_2 = bin_idx_1 + 1
    
    # Agirlik hesaplama 
    weight_2 = bin_indices - bin_idx_1 
    weight_1 = 1.0 - weight_2
    
    # Wrap-around islemi (Dairesellik)
    bin_idx_1 = bin_idx_1 % num_bins
    bin_idx_2 = bin_idx_2 % num_bins
    
    mags = cell_magnitude.flatten()
    
    # Histogrami doldurma (Vektörize işlem)
    np.add.at(histogram, bin_idx_1.flatten(), mags * weight_1.flatten())
    np.add.at(histogram, bin_idx_2.flatten(), mags * weight_2.flatten())
    
    return histogram

def normalize_block(block_vector: np.ndarray, method: str = "L2") -> np.ndarray:
    epsilon = 1e-5 # Sifira bolunmeyi onlemek icin kucuk bir sabit [cite: 110]
    
    if method == "L2":
        # L2-norm: v / sqrt(||v||^2 + epsilon^2)
        return block_vector / np.sqrt(np.sum(block_vector**2) + epsilon**2)
    elif method == "L1":
        # L1-norm: v / (||v|| + epsilon)
        return block_vector / (np.sum(np.abs(block_vector)) + epsilon)
    else:
        return block_vector

def compute_hog_descriptor(image: np.ndarray, cell_size: tuple = (8, 8), block_size: tuple = (2, 2), num_bins: int = 9) -> np.ndarray:
    """
    Görüntüden tam HOG özellik vektörünü hesaplar.
    
    [cite_start]PDF Referansı: Bölüm 3.1 [cite: 100-104]
    
    Args:
        image (np.ndarray): Gri tonlamalı görüntü.
        cell_size (tuple): Hücre boyutu (piksel), örn: (8,8).
        block_size (tuple): Blok boyutu (hücre sayısı), örn: (2,2).
        num_bins (int): Histogram bin sayısı.
        
    Returns:
        np.ndarray: Düzleştirilmiş HOG özellik vektörü.
    """
    # 1. Gradyan Hesaplama
    magnitude, angle = compute_gradients(image)
    
    h_img, w_img = image.shape
    c_h, c_w = cell_size
    b_h, b_w = block_size
    
    # Hucre sayilari
    n_cells_y = h_img // c_h
    n_cells_x = w_img // c_w
    
    # 2. Hücre Histogramlarını Hesapla (Grid of Histograms)
    hist_grid = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)
    
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            y_start = y * c_h
            y_end = y_start + c_h
            x_start = x * c_w
            x_end = x_start + c_w
            
            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ang = angle[y_start:y_end, x_start:x_end]
            
            hist_grid[y, x, :] = create_cell_histogram(cell_mag, cell_ang, num_bins)
            
    # 3. Blok Normalizasyonu ve Vektör Oluşturma (Sliding Window)
    # Blok sayisi hesabi (Slide adimi 1 hücredir)
    n_blocks_y = n_cells_y - b_h + 1
    n_blocks_x = n_cells_x - b_w + 1
    
    hog_vector = []
    
    if n_blocks_y <= 0 or n_blocks_x <= 0:
        print("[WARNING] Görüntü boyutu HOG parametreleri için çok küçük!")
        return np.array([])

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            # Bloğu oluşturan hücreleri al ve düzleştir
            block = hist_grid[y : y + b_h, x : x + b_w, :].flatten()
            
            # Bloğu normalize et
            normalized_block = normalize_block(block, method="L2")
            
            # Sonuç vektörüne ekle
            hog_vector.extend(normalized_block)
            
    return np.array(hog_vector, dtype=np.float32)

def visualize_hog(image: np.ndarray, cell_size: tuple = (8, 8), num_bins: int = 9) -> np.ndarray:
    """
    HOG özelliklerini görselleştirir.
    
    Her hücre için histogram bin değerlerine karşılık gelen yönlerde çizgiler çizer.
    
    Args:
        image (np.ndarray): Orijinal gri tonlamalı görüntü.
        cell_size (tuple): Hücre boyutu.
        num_bins (int): Bin sayısı.
        
    Returns:
        np.ndarray: Görselleştirilmiş HOG görüntüsü.
    """
    # 1. Gradyan ve Histogram Hesapla
    # Not: Görselleştirme için normalize edilmemiş hücre histogramlarını kullanmak
    # yerel detayları görmek için daha iyidir.
    magnitude, angle = compute_gradients(image)
    
    h, w = image.shape
    c_h, c_w = cell_size
    
    n_cells_y = h // c_h
    n_cells_x = w // c_w
    
    # Görselleştirme için boş bir tuval (siyah) veya orijinal görüntünün kopyası
    vis_image = np.zeros((h, w), dtype=np.uint8)
    
    # Radyan cinsinden bin merkez açıları
    bin_width_deg = 180.0 / num_bins
    bin_centers_deg = np.arange(num_bins) * bin_width_deg + (bin_width_deg / 2)
    bin_centers_rad = np.radians(bin_centers_deg)
    
    # Hızlandırma için ön hesaplama: cos ve sin değerleri
    cos_vals = np.cos(bin_centers_rad)
    sin_vals = np.sin(bin_centers_rad)
    
    max_mag = 0 # Normalizasyon için maksimum değeri tutacağız
    
    #tüm histogramları hesaplama
    hist_grid = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)
    
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            y_start, y_end = y * c_h, (y + 1) * c_h
            x_start, x_end = x * c_w, (x + 1) * c_w
            
            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ang = angle[y_start:y_end, x_start:x_end]
            
            hist = create_cell_histogram(cell_mag, cell_ang, num_bins)
            hist_grid[y, x, :] = hist
            
            # Global maksimumu bul (çizgi kalınlığı/uzunluğu ayarı için)
            if np.max(hist) > max_mag:
                max_mag = np.max(hist)

    # 2. Çizgileri Çiz
    # Sıfıra bölünmeyi önle
    if max_mag == 0: max_mag = 1 
    
    scale_factor = 255.0 / max_mag # Çizgi parlaklığı için
    
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            # Hücre merkezi
            cx = x * c_w + c_w // 2
            cy = y * c_h + c_h // 2
            
            cell_hist = hist_grid[y, x, :]
            
            for bin_idx in range(num_bins):
                mag = cell_hist[bin_idx]
                
                # Eğer gradyan çok zayıfsa çizme (gürültüyü azaltır)
                if mag < 0.1 * max_mag:
                    continue
                
                # Çizgi uzunluğu ve parlaklığı büyüklükle orantılı
                strength = mag / max_mag
                line_len = int((min(c_h, c_w) / 2) * strength) # Hücrenin yarısı kadar max uzunluk
                if line_len < 1: line_len = 1
                
                # Çizgi koordinatları
                # (cx, cy) merkezinden +yön ve -yön'e uzatıyoruz
                dx = int(line_len * cos_vals[bin_idx])
                dy = int(line_len * sin_vals[bin_idx])
                
                # Açı 0-180 arası olduğu için dy pozitif (aşağı yönlü) olabilir
                # Görüntü koordinat sisteminde y aşağı artar, ama sinüs matematikseldir.
                # Görsel olarak doğru yönelim için:
                # dy negatife, dx pozitife
                
                p1 = (cx - dx, cy - dy)
                p2 = (cx + dx, cy + dy)
                
                # Çizgi Rengi (Parlaklık)
                intensity = int(255 * strength)
                
                cv2.line(vis_image, p1, p2, (intensity), 1)
                
    return vis_image

# --- DOSYANIN EN ALTINA EKLENECEK GÜNCEL TEST KODU ---

def generate_test_images():
    """
    PDF [cite: 124-128] gereksinimlerine uygun 5 farklı test görüntüsü oluşturur.
    """
    images = []
    names = []
    
    # Boyut
    H, W = 128, 64
    
    # 1. Kare (Basit Geometrik)
    img1 = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(img1, (16, 32), (48, 96), 255, -1)
    images.append(img1); names.append("1_Kare")
    
    # 2. Daire (Basit Geometrik)
    img2 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(img2, (32, 64), 20, 255, -1)
    images.append(img2); names.append("2_Daire")
    
    # 3. Üçgen (Basit Geometrik)
    img3 = np.zeros((H, W), dtype=np.uint8)
    pts = np.array([[32, 30], [10, 90], [54, 90]], np.int32)
    cv2.fillPoly(img3, [pts], 255)
    images.append(img3); names.append("3_Ucgen")
    
    # 4. Yıldız (Kenarları Belirgin Nesne)
    img4 = np.zeros((H, W), dtype=np.uint8)
    # Basit bir yıldız/çarpı şekli
    cv2.line(img4, (10, 10), (54, 118), 255, 5)
    cv2.line(img4, (54, 10), (10, 118), 255, 5)
    images.append(img4); names.append("4_Yildiz")
    
    # 5. İnsan Silüeti (Basit Temsil)
    img5 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(img5, (32, 20), 10, 255, -1) # Baş
    cv2.line(img5, (32, 30), (32, 80), 255, 4) # Gövde
    cv2.line(img5, (32, 40), (10, 60), 255, 4) # Kol Sol
    cv2.line(img5, (32, 40), (54, 60), 255, 4) # Kol Sağ
    cv2.line(img5, (32, 80), (15, 120), 255, 4) # Bacak Sol
    cv2.line(img5, (32, 80), (49, 120), 255, 4) # Bacak Sağ
    images.append(img5); names.append("5_Insan")
    
    return zip(names, images)

if __name__ == "__main__":
    print("--- HOG Problem 1: Kapsamlı Testler ---")
    
    # Test edilecek parametre kombinasyonları 
    # (cell_size, num_bins)
    parameters = [
        ((8, 8), 9),   # Standart
        ((4, 4), 9),   # Daha küçük hücre (Daha detaylı)
        ((8, 8), 12)   # Daha fazla açı (Daha hassas yönelim)
    ]
    
    test_data = list(generate_test_images())
    
    # Sonuçları kaydetmek için klasör kontrolü
    import os
    save_dir = "hog_test_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n[BİLGİ] Test görüntüleri '{save_dir}' klasörüne kaydedilecek.\n")

    for name, img in test_data:
        print(f"\n--- Görüntü: {name} ---")
        
        for (cell_size, n_bins) in parameters:
            print(f"   > Parametreler: Cell={cell_size}, Bins={n_bins}")
            
            # 1. Özellik Vektörü Hesapla ve Boyut Yazdır 
            hog_feat = compute_hog_descriptor(img, cell_size=cell_size, num_bins=n_bins)
            print(f"     Vektör Boyutu: {hog_feat.shape}")
            
            # 2. Görselleştirme 
            vis_hog = visualize_hog(img, cell_size=cell_size, num_bins=n_bins)
            
            # 3. Yan Yana Göster (Orijinal + HOG) 
            # Boyut eşitleme gerekebilir mi? Genellikle aynı boyutta döner ama garanti olsun.
            if vis_hog.shape != img.shape:
                vis_hog = cv2.resize(vis_hog, (img.shape[1], img.shape[0]))
                
            combined = np.hstack((img, vis_hog))
            
            # Kaydet
            filename = f"{save_dir}/{name}_c{cell_size[0]}_b{n_bins}.png"
            cv2.imwrite(filename, combined)
            print(f"     Kaydedildi: {filename}")

    print("\n[BAŞARILI] Tüm test senaryoları tamamlandı.")
    print("Lütfen 'hog_test_results' klasöründeki çıktıları inceleyip rapora ekleyin.")