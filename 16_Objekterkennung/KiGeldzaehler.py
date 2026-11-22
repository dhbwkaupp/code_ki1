"""
KI-Geldzähler: Live-Demo mit der Webcam
=======================================

Erkennt Euro-Münzen im Kamerabild und zeigt die Summe an.

Bedienung im Fenster:
    Regler "conf"   Schwellenwert für die Confidence (in Prozent)
    Regler "iou"    IoU-Schwelle tau für NMS (in Prozent)
    Taste n         NMS ein/aus (YOLO26: One-to-Many- bzw. One-to-One-Head)
    Taste k         zur nächsten Kamera wechseln
    Taste e         Einstellungsdialog der Kamera öffnen (Windows: Fokus, Belichtung, ...)
    Taste s         aktuelles Bild speichern (z. B. für das Live-Testset)
    Taste q         beenden

Aufruf:
    python KiGeldzaehler.py                  # Kamera wird beim Start ausgewählt
    python KiGeldzaehler.py --camera 1       # Kamera direkt vorgeben
    python KiGeldzaehler.py --image testbild.jpg      # ohne Kamera
"""
import argparse
import platform
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

VALUES = {"1ct": 0.01, "2ct": 0.02, "5ct": 0.05, "10ct": 0.10,
         "20ct": 0.20, "50ct": 0.50, "1eur": 1.00, "2eur": 2.00}
WINDOW = "KI-Geldzaehler"
# DirectShow öffnet Kameras unter Windows schneller und zuverlässiger
BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def open_camera(index):
    camera = cv2.VideoCapture(index, BACKEND)
    if camera.isOpened() and camera.read()[0]:
        return camera
    camera.release()
    return None


def configure_camera(camera, width, height):
    """Höhere Auflösung, Autofokus und automatische Belichtung einstellen.
    Ob eine Einstellung wirkt, hängt von Kamera und Treiber ab."""
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # hohe Auflösung bei guter Bildrate
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    # Automatische Belichtung: DirectShow erwartet 1, Linux (V4L2) den Wert 0.75
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if BACKEND == cv2.CAP_DSHOW else 0.75)
    camera.set(cv2.CAP_PROP_AUTO_WB, 1)
    w, h = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Auflösung: {w} x {h}, Autofokus: {camera.get(cv2.CAP_PROP_AUTOFOCUS):.0f}")


def find_cameras(max_index=6):
    """Prüft die Kamera-Indizes 0 bis max_index - 1 und gibt die nutzbaren zurück."""
    cameras = []
    for index in range(max_index):
        camera = open_camera(index)
        if camera is not None:
            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append((index, f"{width} x {height}"))
            camera.release()
    return cameras


def choose_camera():
    print("Suche Kameras ...")
    cameras = find_cameras()
    if not cameras:
        raise SystemExit("Keine Kamera gefunden.")
    for index, resolution in cameras:
        print(f"  [{index}] Kamera {index} ({resolution})")
    if len(cameras) == 1:
        return cameras[0][0]
    available = [index for index, _ in cameras]
    while True:
        answer = input(f"Welche Kamera? [{available[0]}]: ").strip()
        if answer == "":
            return available[0]
        if answer.isdigit() and int(answer) in available:
            return int(answer)
        print("Bitte eine der angezeigten Nummern eingeben.")


def evaluate(model, image, conf, iou, nms):
    r = model.predict(image, conf=conf, iou=iou, nms=nms, agnostic_nms=True, verbose=False)[0]
    total = sum(VALUES.get(r.names[int(c)], 0) for c in r.boxes.cls)
    return r, total


def annotate(r, total, conf, iou, nms, fps=None):
    image = r.plot()
    rows = [f"Summe: {total:.2f} EUR",
              f"Muenzen: {len(r.boxes)}   conf={conf:.2f}   iou={iou:.2f}   NMS={'an' if nms else 'aus'}"]
    if fps is not None:
        rows[-1] += f"   {fps:.0f} FPS"
    cv2.rectangle(image, (0, 0), (image.shape[1], 80), (30, 30, 30), -1)
    cv2.putText(image, rows[0], (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (89, 177, 120), 3)
    cv2.putText(image, rows[1], (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/detect/muenzen/weights/best.pt")
    p.add_argument("--camera", type=int, help="Kamera-Index, ohne Angabe Auswahl beim Start")
    p.add_argument("--image", help="Bilddatei statt Kamera")
    p.add_argument("--width", type=int, default=1920, help="gewünschte Kamerabreite")
    p.add_argument("--height", type=int, default=1080, help="gewünschte Kamerahöhe")
    p.add_argument("--output", help="Ergebnisbild speichern statt anzeigen (nur mit --image)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    a = p.parse_args()

    model = YOLO(a.model)
    nms = True

    if a.image and a.output:                       # ohne Fenster, z. B. zum Testen
        r, total = evaluate(model, a.image, a.conf, a.iou, nms)
        cv2.imwrite(a.output, annotate(r, total, a.conf, a.iou, nms))
        print(f"Summe: {total:.2f} EUR, {len(r.boxes)} Muenzen")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("conf", WINDOW, int(a.conf * 100), 100, lambda v: None)
    cv2.createTrackbar("iou", WINDOW, int(a.iou * 100), 100, lambda v: None)

    camera, camera_index = None, None
    if not a.image:
        camera_index = a.camera if a.camera is not None else choose_camera()
        camera = open_camera(camera_index)
        if camera is None:
            raise SystemExit(f"Kamera {camera_index} lässt sich nicht öffnen.")
        configure_camera(camera, a.width, a.height)
    still_image = cv2.imread(a.image) if a.image else None
    last_time = time.perf_counter()

    while True:
        if camera is not None:
            ok, frame = camera.read()
            if not ok:
                print("Kein Kamerabild")
                break
        else:
            frame = still_image.copy()

        conf = max(cv2.getTrackbarPos("conf", WINDOW), 1) / 100
        iou = max(cv2.getTrackbarPos("iou", WINDOW), 1) / 100
        r, total = evaluate(model, frame, conf, iou, nms)

        current_time = time.perf_counter()
        fps = 1 / max(current_time - last_time, 1e-6)
        last_time = current_time
        cv2.imshow(WINDOW, annotate(r, total, conf, iou, nms, fps))

        key_code = cv2.waitKey(1) & 0xFF
        if key_code == ord("q"):
            break
        if key_code == ord("n"):
            nms = not nms
        if key_code == ord("k") and camera is not None:
            for step in range(1, 7):               # nächste funktionierende Kamera suchen
                next_index = (camera_index + step) % 6
                next_camera = open_camera(next_index) if next_index != camera_index else None
                if next_camera is not None:
                    camera.release()
                    camera, camera_index = next_camera, next_index
                    print("Kamera", camera_index)
                    configure_camera(camera, a.width, a.height)
                    break
        if key_code == ord("e") and camera is not None:
            camera.set(cv2.CAP_PROP_SETTINGS, 1)   # nur unter Windows (DirectShow)
        if key_code == ord("s"):
            name = f"live_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(name, frame)
            print("Gespeichert:", name)

    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
