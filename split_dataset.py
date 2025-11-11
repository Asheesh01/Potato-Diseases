import os, shutil, random
from pathlib import Path

random.seed(42)
src = Path('Dataset')  # your dataset folder name
dst = Path('data')

for split in ['train', 'test']:
    for cls in ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']:
        (dst / split / cls).mkdir(parents=True, exist_ok=True)

for cls in src.iterdir():
    if not cls.is_dir():
        continue
    files = list(cls.glob('*.jpg')) + list(cls.glob('*.png'))
    random.shuffle(files)
    n_train = int(0.8 * len(files))
    for i, f in enumerate(files):
        target = dst / ('train' if i < n_train else 'test') / cls.name / f.name
        shutil.copy(f, target)

print("✅ Dataset split complete! Data stored in 'data/train' and 'data/test'.")
