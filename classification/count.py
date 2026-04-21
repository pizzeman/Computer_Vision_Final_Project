"""
Sheep Counter using Ultralytics YOLO
------------------------------------
Detects and counts sheep in images using a pre-trained YOLO model.
No custom training required — COCO includes the 'sheep' class.

Requirements:
    pip install ultralytics opencv-python

Usage:
    python sheep_counter.py --image path/to/image.jpg
    python sheep_counter.py --image path/to/image.jpg --conf 0.4 --show
"""

import argparse
import sys
from pathlib import Path

import cv2
try:
    from ultralytics import YOLO
except ImportError as import_error:
    raise ImportError(
        "The 'ultralytics' package is not installed for this Python interpreter. "
        "Install it with: python -m pip install ultralytics"
    ) from import_error

def load_model(conf_threshold: float = 0.35) -> YOLO:

    print("[*] Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    model.overrides["conf"] = conf_threshold
    print("[✓] Model loaded successfully.\n")
    return model


def count_sheep(image_path: str, model: YOLO) -> dict:
    """
    Run inference on an image and count sheep detections.

    Args:
        image_path: Path to the input image file.
        model:      Loaded YOLO model.

    Returns:
        dict with keys:
            - count (int): number of sheep detected
            - results: raw Ultralytics Results object
            - image_path (str): original path
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[*] Running inference on: {path.name}")
    conf_threshold = float(model.overrides.get("conf", 0.35))
    prediction_results = model.predict(
        source=str(path),
        conf=conf_threshold,
        verbose=False,
    )
    result = prediction_results[0]
    detections = result.boxes
    sheep_count = len(detections)

    print(f"[✓] Detected {sheep_count} sheep.\n")

    if sheep_count > 0:
        print(f"{'#':<5} {'Confidence':>12}  {'Bounding Box (x1,y1,x2,y2)'}")
        print("-" * 60)
        for index, box in enumerate(detections):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            print(f"{index+1:<5} {confidence:>11.2%}  ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

    return {"count": sheep_count, "results": result, "image_path": str(path)}

def annotate_and_show(result: dict, save: bool = True) -> None:
    """
    Draw bounding boxes and a sheep count label on the image,
    then display it (and optionally save it).
    """
    results = result["results"]
    img_path = result["image_path"]
    count = result["count"]

    # Render YOLO annotations onto the image
    img_bgr = results.plot()

    # Overlay a bold sheep count in the top-left corner
    label = f"Sheep count: {count}"
    font, scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 1.2, 2
    (w, h), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(img_bgr, (10, 10), (10 + w + 10, 10 + h + baseline + 10), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (15, 10 + h + 4), font, scale, (255, 255, 255), thickness)

    if save:
        out_path = Path(img_path).with_name(Path(img_path).stem + "_sheep_count.jpg")
        cv2.imwrite(str(out_path), img_bgr)
        print(f"[✓] Annotated image saved to: {out_path}")

    cv2.imshow("Sheep Counter", img_bgr)
    print("[*] Press any key to close the preview window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Programmatic API — import and call directly from another script
# ---------------------------------------------------------------------------

def count_sheep_in_image(image_path: str, conf_threshold: float = 0.35) -> int:
    """
    Convenience function: load model, count sheep, return the count.

    Example:
        from sheep_counter import count_sheep_in_image
        n = count_sheep_in_image("farm.jpg")
        print(f"There are {n} sheep.")
    """
    model = load_model(conf_threshold)
    result = count_sheep(image_path, model)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count sheep in an image using PyTorch + YOLOv5."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold (0–1). Default: 0.35"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the annotated image in a window after inference."
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
        help="Save the annotated image (default: True)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model(conf_threshold=args.conf)
    result = count_sheep(args.image, model)

    print(f"\n{'='*40}")
    print(f"  Total sheep in image: {result['count']}")
    print(f"{'='*40}\n")

    if args.show:
        annotate_and_show(result, save=args.save)


if __name__ == "__main__":
    main()