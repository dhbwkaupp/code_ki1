"""Eigene Münzfotos mit einem trainierten Modell vorannotieren.

Das Modell schlägt Boxen und Klassen vor, wir korrigieren sie anschließend nur noch
in einem Annotationswerkzeug (z. B. Roboflow). Das ist deutlich schneller als alles
von Hand zu zeichnen.

- Identische Dateien werden nur einmal übernommen.
- Die Fotos werden auf höchstens 1600 Pixel Kantenlänge verkleinert.
- Ergebnis im YOLO-Format: images/, labels/, data.yaml, classes.txt
- Annotiert werden alle acht Münzsorten. Daraus lässt sich später sowohl der Datensatz
  mit acht als auch der mit fünf Klassen erzeugen.
- Gegenstände, die keine Euro-Münzen sind (Einkaufswagenchips ...), erhalten keine Box.

Aufruf:
    python Vorannotieren.py --source "../Datensätze/eigene fotos"
"""
import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

CLASSES = ["1ct", "2ct", "5ct", "10ct", "20ct", "50ct", "1eur", "2eur"]
MAX_SIZE = 1600


def pre_annotate(model_path, source, target, conf, imgsz):
    model = YOLO(model_path)
    source, target = Path(source), Path(target)
    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "labels").mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    num_boxes, seen = 0, set()
    for i, file in enumerate(files, 1):
        digest = hashlib.md5(file.read_bytes()).hexdigest()
        if digest in seen:                    # identische Kopie, z. B. "foto 1.jpg"
            print(f"{i:3d}/{len(files)}  {file.name}: Duplikat, übersprungen")
            continue
        seen.add(digest)
        # Handyfotos: EXIF-Drehung anwenden, dann verkleinern
        image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        image.thumbnail((MAX_SIZE, MAX_SIZE))
        image_path = target / "images" / f"{file.stem}.jpg"
        image.save(image_path, quality=90)

        # größeres imgsz findet kleine Münzen auf den hochaufgelösten Fotos besser
        r = model.predict(image_path, conf=conf, imgsz=imgsz, verbose=False)[0]
        rows = []
        for cls, box in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
            class_id = CLASSES.index(r.names[int(cls)])   # Modell muss alle acht Klassen kennen
            rows.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in box))
        (target / "labels" / f"{file.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        num_boxes += len(rows)
        print(f"{i:3d}/{len(files)}  {file.name}: {len(rows)} Boxen")

    (target / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    (target / "data.yaml").write_text(
        "train: images\nval: images\n\nnames:\n"
        + "".join(f"  {i}: {k}\n" for i, k in enumerate(CLASSES)), encoding="utf-8")
    print(f"{len(seen)} Bilder, {num_boxes} Boxen vorgeschlagen -> {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="runs/detect/muenzen-2/weights/best.pt")
    parser.add_argument("--source", default="../Datensätze/eigene fotos")
    parser.add_argument("--target", default="../Datensätze/eigene_fotos_vorannotiert")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    pre_annotate(args.model, args.source, args.target, args.conf, args.imgsz)
