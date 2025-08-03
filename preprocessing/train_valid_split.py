# SPLITTING TRAINING - VALIDATION IN 80/20

#Problema a linea 32 (sar_images non contiene nulla)

import os
import glob
import shutil
import random

# Percorsi di input
base_dir = "..\..\CAFFE"

sar_dir = os.path.join(base_dir, "sar_images", "train")
front_dir = os.path.join(base_dir, "fronts", "train")
zone_dir = os.path.join(base_dir, "zones", "train")

# Percorsi per la validazione
sar_valid_dir = os.path.join(base_dir, "sar_images", "valid")
front_valid_dir = os.path.join(base_dir, "fronts", "valid")
zone_valid_dir = os.path.join(base_dir, "zones", "valid")

print(sar_dir)



# Crea cartelle valid se non esistono
os.makedirs(sar_valid_dir, exist_ok=True)
os.makedirs(front_valid_dir, exist_ok=True)
os.makedirs(zone_valid_dir, exist_ok=True)

# Recupera immagini SAR
sar_images = sorted(glob.glob(os.path.join(sar_dir, '*.png')))

#print(len(sar_images))

# Split 80/20
random.seed(42)
n_valid = int(0.2 * len(sar_images))

#print(n_valid)
valid_images = random.sample(sar_images, n_valid)

#print(len(valid_images))

for sar_path in valid_images:
    filename = os.path.basename(sar_path)

    # Percorsi associati
    front_path = os.path.join(front_dir, filename)
    zone_path = os.path.join(zone_dir, filename)

    # Nuove destinazioni
    shutil.move(sar_path, os.path.join(sar_valid_dir, filename))
    shutil.move(front_path, os.path.join(front_valid_dir, filename))
    shutil.move(zone_path, os.path.join(zone_valid_dir, filename))

print(f"✔ Split completato: {n_valid} immagini spostate in 'valid/'.")
