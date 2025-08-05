# PREPROCESSING STEP 0:

# SPLITTING TRAINING - VALIDATION IN 80/20

# ONLY RUN IT IF YOU HAVEN'T ALREADY SPLIT DATASET INTO TRAIN/VALID SETS

import shutil
import random
from pathlib import Path

# Percorsi di input
base_dir = Path(__file__).resolve().parents[2] / "CAFFE"

sar_dir = base_dir / "sar_images" / "train"
zone_dir = base_dir / "zones" / "train"
front_dir = base_dir / "fronts" / "train"


# Percorsi per la validazione
sar_valid_dir = base_dir / "sar_images" / "valid"
zone_valid_dir = base_dir / "zones" / "valid"
front_valid_dir = base_dir / "fronts" / "valid"



# Crea cartelle valid se non esistono
sar_valid_dir.mkdir(parents=True, exist_ok=True)
front_valid_dir.mkdir(parents=True, exist_ok=True)
zone_valid_dir.mkdir(parents=True, exist_ok=True)

# Recupera immagini SAR
sar_images = sorted(sar_dir.glob("*.png"))

#print(len(sar_images))

# Split 80/20
random.seed(42)
n_valid = int(0.2 * len(sar_images))

#print(n_valid)
valid_images = random.sample(sar_images, n_valid)

#print(len(valid_images))

for sar_path in valid_images:
    filename = sar_path.name
    stem = sar_path.stem

    # Percorsi associati: nelle cartelle `fronts` e `zones`
    # i file hanno rispettivamente i suffissi `_front` e `_zones`
    front_path = front_dir / f"{stem}_front{sar_path.suffix}"
    zone_path = zone_dir / f"{stem}_zones{sar_path.suffix}"

    # Skip se mancano i file associati
    if not front_path.exists() or not zone_path.exists():
        missing = []
        if not front_path.exists():
            missing.append("front")
        if not zone_path.exists():
            missing.append("zone")
        print(f"⚠ Missing images ({', '.join(missing)}) for {filename}, skipping.")
        continue

    # Nuove destinazioni
    shutil.move(sar_path, sar_valid_dir / filename)
    shutil.move(front_path, front_valid_dir / front_path.name)
    shutil.move(zone_path, zone_valid_dir / zone_path.name)

print(f"✔ Split completed: {n_valid} images moved in 'valid/'.")