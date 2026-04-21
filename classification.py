"""
Animal Classifier using Ultralytics YOLO + SVM
-----------------------------------------------
YOLO detects animals, SVM classifies each detection. Compare SVM to ResNet

Requirements:
    pip install tqdm opencv-python numpy torch torchvision joblib scikit-learn ultralytics

Usage:
    1. Example command to run in terminal (includes training of ResNet):
        python classification.py --image random_images/multiple_sheep.jpg --train --labeled_dir "path/to/train_data" --model resnet --show
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm
from count import annotate_and_show
from count import count_sheep

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    from ultralytics import YOLO
except ImportError as import_error:
    raise ImportError(
        "The 'ultralytics' package is not installed. "
        "Install it with: python -m pip install ultralytics"
    ) from import_error


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(conf_threshold: float = 0.35) -> YOLO:
    print("[*] Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    model.overrides["conf"] = conf_threshold
    print("[✓] Model loaded successfully.\n")
    return model

# ---------------------------------------------------------------------------
# Feature extraction (from YOLO backbone)
# ---------------------------------------------------------------------------

def extract_features(crop: np.ndarray, model: YOLO) -> np.ndarray:
    """Extract feature embeddings using YOLO's embed method."""
    resized = cv2.resize(crop, (640, 640))
    results = model.embed(source=resized, verbose=False)
    return results[0].cpu().numpy().flatten()

# ---------------------------------------------------------------------------
# SVM training and loading
# ---------------------------------------------------------------------------

def build_dataset(labeled_dir: str, model: YOLO):
    """
    Build feature matrix from labeled crops.

    labeled_dir/
        frog/
            img1.jpg ...
        sheep/
            img1.jpg ...
    """
    X, y = [], []
    classes = sorted(os.listdir(labeled_dir))

    for label, cls in enumerate(classes):
        cls_dir = os.path.join(labeled_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        print(f"[*] Extracting features for class: {cls}")
        files = [f for f in os.listdir(cls_dir) 
                if cv2.imread(os.path.join(cls_dir, f)) is not None]
        for fname in tqdm(files, desc=f"[*] {cls}", unit="img"):
            img = cv2.imread(os.path.join(cls_dir, fname))
            features = extract_features(img, model)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y), classes

def train_svm(labeled_dir: str, model: YOLO, save_path: str = "svm_classifier.pkl"):
    """Train SVM on labeled crops and save to disk."""
    X, y, classes = build_dataset(labeled_dir, model)

    # 70/15/15 split train, val, test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    print("[*] Training SVM...")
    clf.fit(X_train, y_train)
    print(f"Train accuracy: {clf.score(X_train, y_train):.2%}")
    print(f"Test  accuracy: {clf.score(X_test, y_test):.2%}")
    print(classification_report(y_test, clf.predict(X_test), target_names=classes))

    # Save classifier and class names together
    joblib.dump({"clf": clf, "classes": classes}, save_path)
    print(f"[✓] SVM saved to: {save_path}")
    return clf, classes


def load_svm(path: str = "svm_classifier.pkl"):
    """Load SVM classifier and class names from disk."""
    data = joblib.load(path)
    return data["clf"], data["classes"]

# ---------------------------------------------------------------------------
# ResNet Training + Loading
# ---------------------------------------------------------------------------
RESNET_TRANSFORMS = {
    "train": T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}
 
 
class CropDataset(Dataset):
    """Loads labeled image crops from a directory tree."""
 
    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
 
 
def _collect_image_paths(labeled_dir: str):
    """Walk labeled_dir and return (paths, int_labels, class_names)."""
    classes = sorted([
        d for d in os.listdir(labeled_dir)
        if os.path.isdir(os.path.join(labeled_dir, d))
    ])
    paths, labels = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(labeled_dir, cls)
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if cv2.imread(fpath) is not None:
                paths.append(fpath)
                labels.append(label)
    return paths, labels, classes
 
 
def build_resnet(num_classes: int) -> nn.Module:
    """ResNet-50 with a fine-tuned classification head."""
    resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
 
    # Freeze backbone, only train the final FC layer initially
    for param in resnet.parameters():
        param.requires_grad = False
 
    resnet.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(resnet.fc.in_features, num_classes),
    )
    return resnet
 
def test_resnet(model: nn.Module, labeled_dir: str):
    """Evaluate a loaded ResNet model on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    paths, labels, classes = _collect_image_paths(labeled_dir)
    idx = list(range(len(paths)))
    _, temp_idx = train_test_split(idx, test_size=0.3, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    _, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels)

    test_paths  = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_dataset = CropDataset(test_paths, test_labels, transform=RESNET_TRANSFORMS["val"])
    test_loader  = DataLoader(test_dataset, batch_size=32)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls.tolist())

    print(classification_report(all_labels, all_preds, target_names=classes))
 
def train_resnet(
    labeled_dir: str,
    save_path: str = "resnet_classifier.pth",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """Fine-tune ResNet-50 on labeled crops and save to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training ResNet on: {device}")
 
    paths, labels, classes = _collect_image_paths(labeled_dir)
 
    # 70/15/15 split
    idx = list(range(len(paths)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels)
 
    def subset(indices, split):
        p = [paths[i] for i in indices]
        l = [labels[i] for i in indices]
        return CropDataset(p, l, transform=RESNET_TRANSFORMS[split])
 
    loaders = {
        "train": DataLoader(subset(train_idx, "train"), batch_size=batch_size, shuffle=True),
        "val":   DataLoader(subset(val_idx,   "val"),   batch_size=batch_size),
        "test":  DataLoader(subset(test_idx,  "val"),   batch_size=batch_size),
    }
 
    resnet = build_resnet(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
 
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        resnet.train()
        running_loss = 0.0
        for imgs, lbls in tqdm(loaders["train"], desc=f"Epoch {epoch}/{epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(resnet(imgs), lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
 
        # Validation
        resnet.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in loaders["val"]:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = resnet(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        val_acc = correct / total
        print(f"  Loss: {running_loss/len(loaders['train']):.4f} | Val Acc: {val_acc:.2%}")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": resnet.state_dict(), "classes": classes}, save_path)
            print(f"  [✓] Best model saved (val_acc={val_acc:.2%})")
 
    # Final test evaluation
    resnet.load_state_dict(torch.load(save_path)["model"])
    resnet.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loaders["test"]:
            imgs = imgs.to(device)
            preds = resnet(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls.tolist())
    print(classification_report(all_labels, all_preds, target_names=classes))
    print(f"[✓] ResNet saved to: {save_path}")
    return resnet, classes


def load_resnet(path: str = "resnet_classifier.pth"):
    """Load fine-tuned ResNet and class names from disk."""
    data = torch.load(path, map_location="cpu")
    classes = data["classes"]
    resnet = build_resnet(num_classes=len(classes))
    resnet.load_state_dict(data["model"])
    resnet.eval()
    return resnet, classes

# ---------------------------------------------------------------------------
# Detection + classification
# ---------------------------------------------------------------------------

def classify_crop_resnet(crop: np.ndarray, resnet: nn.Module, classes: list, device: torch.device):
    """Run a single crop through the ResNet classifier."""
    tensor = RESNET_TRANSFORMS["val"](cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = resnet(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
    return classes[pred], probs[0, pred].item()
 
 
def detect_and_classify(
    image_path: str,
    model: YOLO,
    clf=None,   
    classes=None,
    resnet=None,
    resnet_classes=None,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    Run YOLO detection, classify each crop with SVM or ResNet.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
 
    print(f"[*] Running inference on: {path.name}")
    prediction_results = model.predict(source=str(path), verbose=False)
    result = prediction_results[0]
    detections = result.boxes
    image = cv2.imread(str(path))
 
    det_count = len(detections)
    print(f"[✓] Detected {det_count} object(s).\n")
 
    labels = []
 
    if det_count > 0:
        print(f"{'#':<5} {'Confidence':>12}  {'Bounding Box (x1,y1,x2,y2)':<30} {'Class'}")
        print("-" * 75)
 
        for index, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            crop = image[y1:y2, x1:x2]
 
            if resnet is not None:
                # ResNet classification
                label, prob = classify_crop_resnet(crop, resnet, resnet_classes, device)
                label_str = f"{label} ({prob:.1%})"
 
            elif clf is not None:
                # SVM classification
                features = extract_features(crop, model).reshape(1, -1)
                label = classes[clf.predict(features)[0]]
                prob = clf.predict_proba(features).max()
                label_str = f"{label} ({prob:.1%})"
 
            else:
                # Fall back to raw YOLO class name
                label = result.names[int(box.cls[0])]
                label_str = label
 
            labels.append(label)
            print(f"{index+1:<5} {confidence:>11.2%}  ({x1}, {y1}, {x2}, {y2}){'':10} {label_str}")
 
    return {
        "count": det_count,
        "labels": labels,
        "results": result,
        "image_path": str(path),
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and classify animals using YOLO + SVM."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", required=True, choices=["svm", "resnet"])
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (0-1). Default: 0.35")
    parser.add_argument("--show", action="store_true",
                        help="Display the annotated image after inference.")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save the annotated image.")

    parser.add_argument("--epochs", type=int, default=5,
                    help="Training epochs (ResNet only). Default: 5")
    parser.add_argument("--confusion", action="store_true",
                        help="Generate confusion matrix after training (ResNet only).")
    parser.add_argument("--train", action="store_true",
                        help="Train model on labeled_dir before running inference.")
    parser.add_argument("--labeled_dir", type=str,
                        help="Path to labeled crop directory (required if --train).")
    parser.add_argument("--model_path", type=str,
                        help="Path to save/load SVM model. Default: choose either svm or resnet")


    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    if device.type == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    model = load_model(conf_threshold=args.conf)

    clf = classes = resnet = resnet_classes = None

    model_type = args.model
    model_path = args.model_path

    default_paths = {"svm": "svm_classifier.pkl", "resnet": "resnet_classifier.pth"}
    model_path = args.model_path or default_paths[model_type]

    if not model_type:
        raise ValueError("--model is required for running (either svm or resnet)")
    
    # Train Model if requested
    if args.train:
        if not args.labeled_dir:
            raise ValueError("--labeled_dir is required when using --train")
        if model_type == "svm":
            clf, classes = train_svm(args.labeled_dir, model, save_path=args.model_path)
        elif model_type == "resnet":
            resnet, resnet_classes = train_resnet(
                args.labeled_dir,
                save_path=model_path,
                epochs=args.epochs,
            )
            resnet = resnet.to(device)
    elif Path(model_path).exists():
        if model_type == "svm":
            print(f"[*] Loading SVM from: {model_path}")
            clf, classes = load_svm(model_path)
 
        elif model_type == "resnet":
            print(f"[*] Loading ResNet from: {model_path}")
            resnet, resnet_classes = load_resnet(model_path)
            resnet = resnet.to(device)

    result = detect_and_classify(
        args.image, model,
        clf=clf, classes=classes,
        resnet=resnet, resnet_classes=resnet_classes,
        device=device,
    )
    print(f"\n{'='*40}")
    print(f"  Total detections: {result['count']}")
    if result["labels"]:
        print(f"  Animals: {', '.join(result['labels'])}")
    print(f"{'='*40}\n")

    if args.confusion and model_type == "resnet" and resnet is not None:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.model_selection import GroupShuffleSplit
        if not args.labeled_dir:
            raise ValueError("--labeled_dir is required when using --confusion")

        print("[*] Generating confusion matrix...")
        paths, labels, _ = _collect_image_paths(args.labeled_dir)
        video_ids = [os.path.basename(p).split("_t")[0] for p in paths]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        _, temp_idx = next(gss.split(paths, labels, groups=video_ids))
        temp_groups = [video_ids[i] for i in temp_idx]
        temp_labels = [labels[i] for i in temp_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        _, test_idx = next(gss2.split(temp_idx, temp_labels, groups=temp_groups))

        test_paths  = [paths[i]  for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        test_loader = DataLoader(
            CropDataset(test_paths, test_labels, transform=RESNET_TRANSFORMS["val"]),
            batch_size=64, num_workers=2, pin_memory=True,
        )

        resnet.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(device)
                with torch.amp.autocast("cuda"):
                    preds = resnet(imgs).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(lbls.tolist())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=resnet_classes,
            yticklabels=resnet_classes,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()
        print("[✓] Confusion matrix saved to confusion_matrix.png")

    if args.show:
        annotate_and_show(result, save=args.save)



if __name__ == "__main__":
    main()