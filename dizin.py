import os
from pathlib import Path

def create_directory(path: Path):
    """
    Belirtilen dizini oluşturur. Dizin varsa hata vermez.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Dizin olusturuldu veya zaten mevcut: {path}")
    except OSError as e:
        print(f"[ERROR] Dizin olusturulurken hata: {path}. Hata: {e}")

def create_file(path: Path, content: str = ""):
    """
    Belirtilen dosyayı oluşturur ve içine başlangıç içeriğini yazar.
    Dosya zaten varsa üzerine yazmaz.
    """
    if not path.exists():
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[INFO] Dosya olusturuldu: {path}")
        except OSError as e:
            print(f"[ERROR] Dosya olusturulurken hata: {path}. Hata: {e}")
    else:
        print(f"[SKIP] Dosya zaten mevcut: {path}")

def main():
    # PDF [cite: 172] uyarinca ad-soyad formatinda kok dizin
    root_dir_name = "aziz-deniz-akmermer-hog-odev"
    root_path = Path(root_dir_name)

    # PDF  referans alinarak hazirlanan dizin yapisi
    directories = [
        root_path / "src",
        root_path / "data" / "test_images",
        root_path / "data" / "training_set",
        root_path / "data" / "results",
        root_path / "models",
        root_path / "report" / "figures",
        root_path / "notebooks"
    ]

    # Olusturulacak dosyalar ve baslangic icerikleri
    # requirements.txt icerigi PDF [cite: 266-269] referans alinarak hazirlanmistir.
    requirements_content = (
        "opencv-python\n"
        "numpy\n"
        "scikit-learn\n"
        "scikit-image\n"
        "matplotlib\n"
        "seaborn\n"
        "pillow\n"
        "joblib\n"
        "jupyter\n"
    )

    # Minimal gecerli bir .ipynb JSON yapisi
    jupyter_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'

    files = {
        root_path / "README.md": f"# HOG Odev - Aziz Deniz Akmermer (220212037)\n\n## Kurulum\n\n```bash\npip install -r requirements.txt\n```\n",
        root_path / "requirements.txt": requirements_content,
        root_path / "src" / "hog_implementation.py": "\"\"\"\nProblem 1: HOG Implementasyonu\n\"\"\"\nimport numpy as np\nimport cv2\n",
        root_path / "src" / "object_detection.py": "\"\"\"\nProblem 2: Nesne Tespiti\n\"\"\"\nimport cv2\nimport numpy as np\n",
        root_path / "src" / "classification.py": "\"\"\"\nProblem 3: Siniflandirma ve Karsilastirma\n\"\"\"\nimport sklearn\n",
        root_path / "src" / "utils.py": "\"\"\"\nYardimci fonksiyonlar ve araclar\n\"\"\"\n",
        root_path / "notebooks" / "analysis.ipynb": jupyter_content,
        # Asagidaki binary/placeholder dosyalar yapiyi tamamlama amaclidir
        root_path / "models" / "trained_classifier.pkl": "", 
        root_path / "report" / "report.pdf": "" 
    }

    print("--- Proje Yapisi Olusturuluyor ---")
    
    # Dizinleri olustur
    create_directory(root_path)
    for directory in directories:
        create_directory(directory)

    # Dosyalari olustur
    for file_path, content in files.items():
        create_file(file_path, content)

    print("\n--- Kurulum Tamamlandi ---")
    print(f"Proje dizini: {root_path.absolute()}")

if __name__ == "__main__":
    main()