from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil
import uuid
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / "data" / "processed"
VERIFIED = PROJECT_ROOT / "data" / "verified"
PROCESSED.mkdir(parents=True, exist_ok=True)
VERIFIED.mkdir(parents=True, exist_ok=True)

# create a few synthetic 64x64 character crops
chars = ['a', 'b', 'F', 'H', 'z']
for i, ch in enumerate(chars, start=1):
    img = Image.new('L', (64,64), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 40)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = font.getsize(ch)
    draw.text(((64-w)/2,(64-h)/2), ch, fill=0, font=font)
    path = PROCESSED / f'char_{i:04d}.png'
    img.save(path)
    print('Wrote', path)

# Run the app logic to predict and persist
from models.cnn_classifier import load_model, predict_character, CLASS_INDEX_TO_CHAR
from app.app import _load_image_from_path, _prepare_prediction_character, _build_character_database_from_verified_labels

model = None
try:
    model = load_model()
    print('Loaded model')
except Exception as e:
    print('Model load failed:', e)

# batch predict
paths = sorted(PROCESSED.glob('char_*.png'))
imgs = []
for p in paths:
    im = _load_image_from_path(p)
    imgs.append(_prepare_prediction_character(im))

if model is not None and imgs:
    X = np.stack([img if img.ndim==3 else img[:,:,np.newaxis] for img in imgs], axis=0).astype(np.float32)
    if X.max()>1.0:
        X /= 255.0
    probs = model.predict(X, verbose=0)
    for p,prob in zip(paths, probs):
        pred_idx = int(np.argmax(prob))
        conf = float(np.max(prob))
        pred = CLASS_INDEX_TO_CHAR[pred_idx]
        print(p.name, '->', pred, f'({conf*100:.1f}%)')

# Simulate corrections: for low confidence (<75%) set to 'a', else keep prediction
corrections = {}
if model is not None and imgs:
    for p,prob in zip(paths, probs):
        pred_idx = int(np.argmax(prob))
        conf = float(np.max(prob))
        pred = CLASS_INDEX_TO_CHAR[pred_idx]
        corrected = pred if conf>=0.75 else 'a'
        corrections[str(p)] = corrected

# Persist corrected files to data/verified/<label>/
VERIFIED.mkdir(parents=True, exist_ok=True)
moved = 0
for src_text, label in corrections.items():
    src = Path(src_text)
    if not src.exists():
        continue
    dest_dir = VERIFIED / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / f"{src.stem}_{uuid.uuid4().hex}{src.suffix}"
    shutil.move(str(src), str(dest))
    moved += 1
    print(f'Moved {src.name} -> {dest}')

# Build DB from verified
verified_map = {str(p): p.parent.name for p in VERIFIED.rglob('*.png')}
char_db = _build_character_database_from_verified_labels(verified_map)
print('Verified DB labels:', list(char_db.keys()))
print('Moved count:', moved)
