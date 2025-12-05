import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[OLUŞTURULDU] {path}")
    else:
        print(f"[MEVCUT] {path}")

# Kök dizin
base = "data"

# 1. İNSAN TESPİTİ İÇİN KLASÖRLER
create_dir(os.path.join(base, "human_detection", "test_images")) # İnsan test fotoları buraya
create_dir(os.path.join(base, "human_detection", "results"))     # İnsan sonuçları buraya

# 2. ARABA TESPİTİ İÇİN KLASÖRLER
create_dir(os.path.join(base, "car_detection", "training_set", "positive")) 
create_dir(os.path.join(base, "car_detection", "training_set", "negative")) 
create_dir(os.path.join(base, "car_detection", "test_images"))              # Araba test fotoları buraya
create_dir(os.path.join(base, "car_detection", "results"))                  

print("\n--- KLASÖR YAPISI DÜZENLENDİ ---")
print("Lütfen dosyalarınızı manuel olarak yeni klasörlere taşıyın:")
print("1. İnsan fotolarını -> data/human_detection/test_images içine")
print("2. Araba fotolarını -> data/car_detection/test_images içine")