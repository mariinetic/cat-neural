"""Prepara um dataset binário para o CatNeural.

Positivos: imagens do Cat Dataset em backend/dataset/raw (ou backend/dataset).
Negativos: classes não-cat do CIFAR-10, baixadas pelo TensorFlow/Keras.

O resultado fica em backend/dataset/train e backend/dataset/validation,
compatível com model/train.py.
"""
from pathlib import Path
import shutil
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset"
PREPARED = ROOT / "dataset" / "prepared"
SEED = 42
NEGATIVE_LIMIT = 3000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(folder: Path):
    return [path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and "prepared" not in path.parts]


def copy_split(paths, label: str):
    split_at = max(1, int(len(paths) * 0.8))
    for split, subset in (("train", paths[:split_at]), ("validation", paths[split_at:])):
        target = PREPARED / split / label
        target.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(subset):
            destination = target / f"{label}_{index:05d}{source.suffix.lower()}"
            try:
                with Image.open(source) as image:
                    image.convert("RGB").save(destination, quality=92)
            except (OSError, ValueError):
                continue


def prepare_cats():
    candidates = image_files(RAW)
    if not candidates:
        raise SystemExit("Nenhuma imagem de gato encontrada em backend/dataset.")
    copy_split(sorted(candidates), "cat")
    return len(candidates)


def prepare_negatives():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    images = list(zip(x_train, y_train.reshape(-1))) + list(zip(x_test, y_test.reshape(-1)))
    # CIFAR-10: cat = 3. Todas as demais classes são exemplos negativos.
    negatives = [(image, label) for image, label in images if int(label) != 3][:NEGATIVE_LIMIT]
    split_at = max(1, int(len(negatives) * 0.8))
    for split, subset in (("train", negatives[:split_at]), ("validation", negatives[split_at:])):
        target = PREPARED / split / "not_cat"
        target.mkdir(parents=True, exist_ok=True)
        for index, (array, _) in enumerate(subset):
            Image.fromarray(array).resize((224, 224)).save(target / f"negative_{index:05d}.jpg", quality=90)
    return len(negatives)


if __name__ == "__main__":
    if PREPARED.exists():
        shutil.rmtree(PREPARED)
    cats = prepare_cats()
    negatives = prepare_negatives()
    print(f"Dataset preparado: {cats} gatos e {negatives} negativos em {PREPARED}")
