import os
from time import time
import pickle
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F

from limb_tracking.limb_tracking import infer

METADATA_FILENAME = "/Users/cwise/Downloads/AR_metadata.xlsx"
VIDEO_DIRECTORY = "/Users/cwise/Downloads/image"
ANIMALS = ["horse", "frog", "lion", "common crane"]
BEHAVIORS = ["running", "fighting", "drinking", "sleeping", "playing"]
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.2
FRAME_SIZE = (224, 224)
FRAMES_PER_VIDEO = 16
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 20
NUM_LIMBS = 23
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
USE_CACHE = True
FORCE_RECOMPUTE = False


def sanitize(s: str) -> str:
    return s.strip().lower().replace(",", "").replace("[","").replace("]","").replace("'","").replace('"',"")


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def load_cache(name: str):
    path = _cache_path(name)
    if not USE_CACHE or FORCE_RECOMPUTE or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cache(name: str, obj) -> None:
    if not USE_CACHE:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_or_build_cached(name: str, builder):
    cached = load_cache(name)
    if cached is not None:
        print(f"Loaded cache: {name}")
        return cached
    value = builder()
    save_cache(name, value)
    print(f"Saved cache: {name}")
    return value

def get_video_ids() -> tuple:
    metadata = pd.read_excel(METADATA_FILENAME)
    video_ids = list()
    for row in metadata.itertuples():
        if "," in row.list_animal: # multiple animals in the video
            continue
        animal = sanitize(row.list_animal)
        if animal not in ANIMALS:
            continue
        actions = ast.literal_eval(row.list_animal_action)
        if not actions:
            continue
        animal_action = actions[0]
        if len(animal_action) < 2:
            continue
        behavior = sanitize(animal_action[1])
        if behavior not in BEHAVIORS:
            continue
        video_ids.append((row.video_id, animal, behavior))
    return video_ids

def split_video_ids(video_ids: list) -> tuple:
    n_test = int(len(video_ids) * TEST_RATIO)
    n_validation = int(len(video_ids) * VALIDATION_RATIO)
    validation_test_ids = video_ids[:n_validation + n_test]
    test_video_ids = video_ids[n_validation:n_validation + n_test]
    train_video_ids = video_ids[n_validation + n_test:]
    return train_video_ids, validation_test_ids, test_video_ids


def get_video_frames(video_id: str) -> list:
    frames = list()
    i = 1
    while True:
        filename = f"{VIDEO_DIRECTORY}/{video_id}/{video_id}_t{i:06d}.jpg"
        if not os.path.exists(filename):
            break
        frames.append(filename)
        i += 1
    return frames

def run_limb_tracking(frame_paths: list, animal: str) -> np.ndarray:
    # array dims: (T, L, 2) where T is number of frames, L is number of limbs, and 2 is (x,y) coordinates
    out = list()
    for frame_path in frame_paths:
        poses = infer(
            model_name="hrnet",
            checkpoint="limb_tracking/results/hrnet_best.pth",
            image_path=frame_path,
        )
        poses = np.asarray(poses, dtype=np.float32)

        # If multiple animals are returned, keep one pose track for this frame.
        if poses.ndim == 3:
            poses = poses[0]

        if poses.ndim != 2 or poses.shape[0] == 0:
            continue

        standardized = np.zeros((NUM_LIMBS, 2), dtype=np.float32)
        n = min(NUM_LIMBS, poses.shape[0])
        standardized[:n, :] = poses[:n, :2]
        out.append(standardized)

    if not out:
        return np.empty((0, NUM_LIMBS, 2), dtype=np.float32)

    return np.asarray(out, dtype=np.float32)

def normalize_frames(frames: np.ndarray, animal: str, behavior: str) -> np.ndarray:
    if len(frames) == 0:
        # Fallback so downstream normalization and batching can proceed.
        frames = np.zeros((FRAMES_PER_VIDEO, NUM_LIMBS, 2), dtype=np.float32)

    # normalize the number of frames to FRAMES_PER_VIDEO by sampling or padding
    if len(frames) > FRAMES_PER_VIDEO: # to many frames
        indices = np.linspace(0, len(frames) - 1, FRAMES_PER_VIDEO).astype(int)
        sampled_frames = [frames[i] for i in indices]
    elif len(frames) < FRAMES_PER_VIDEO: # not enough frames
        pad_count = FRAMES_PER_VIDEO - len(frames)
        pad = np.repeat(frames[-1][None, :, :], pad_count, axis=0)
        sampled_frames = np.concatenate([frames, pad], axis=0)
    else: # just right
        sampled_frames = frames
    frames = np.asarray(sampled_frames)
    # frames is now fixed-length with shape (T, L, 2)

    # center the limbs
    centers = frames[:, 19, :]  # (T, 2)
    frames = frames - centers[:, None, :]  # (T, L, 2) after broadcasting the center point

    # scale the limbs
    head, tail = 0, 16 # head mid top and tail base
    scales = np.linalg.norm(frames[:, head, :] - frames[:, tail, :], axis=1)
    scale = np.mean(scales)
    scale = max(scale, 1e-6)
    frames = frames / scale

    # also add velocity
    velocity = frames[1:] - frames[:-1]  # (T - 1, L, 2)
    velocity = np.concatenate([velocity, velocity[-1:]], axis=0)
    frames = np.concatenate([frames, velocity], axis=2)  # (T, L, 4) after appending dx/dy

    # add one-hot encoding for animal and behavior
    animal_onehot = list_to_onehot(ANIMALS, animal)  # (len(ANIMALS),)
    behavior_onehot = list_to_onehot(BEHAVIORS, behavior)  # (len(BEHAVIORS),)
    onehot = np.concatenate([animal_onehot, behavior_onehot])  # (len(ANIMALS) + len(BEHAVIORS),)
    onehot = np.tile(onehot, (frames.shape[0], 1))  # (T, len(ANIMALS) + len(BEHAVIORS))
    onehot = np.repeat(onehot[:, None, :], frames.shape[1], axis=1)  # (T, L, len(ANIMALS) + len(BEHAVIORS))
    frames = np.concatenate([frames, onehot], axis=2)  # (T, L, 4 + len(ANIMALS) + len(BEHAVIORS)) after appending metadata

    return frames


def list_to_onehot(l: list, item: str) -> np.ndarray:
    onehot = np.zeros(len(l), dtype=np.float32)
    if item in l:
        idx = l.index(item)
        onehot[idx] = 1.0
    else:
        raise ValueError(f"Item '{item}' not found in list {l}")
    return onehot


def flatten_frame_features(frames: np.ndarray) -> np.ndarray:
    # reshape from (T, L, F) to (T, L*F) so BehaviorCNN receives 3D batches (B, T, F_total)
    return frames.reshape(frames.shape[0], -1)



class BehaviorCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, T, F)
        x = x.permute(0, 2, 1)  # → (batch, F, T)

        x = F.relu(self.conv1(x))  # (batch, 64, T)
        x = F.relu(self.conv2(x))  # (batch, 128, T)
        x = F.relu(self.conv3(x))  # (batch, 128, T)

        x = self.pool(x)        # → (batch, 128, 1)
        x = x.squeeze(-1)       # → (batch, 128)

        x = self.fc(x)          # → (batch, num_classes)
        return x
    

def get_device():
    """Gets the best available device for torch."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_model(train_data: list, validation_data: list):
    device = get_device()
    input_dim = train_data[0][0].shape[1]
    model = BehaviorCNN(input_dim=input_dim, num_classes=len(BEHAVIORS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_acc_history = []
    val_acc_history = []
    for epoch in range(EPOCHS):
        model.train()
        np.random.shuffle(train_data)
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            frames_batch = [item[0] for item in batch]
            labels_batch = [item[2] for item in batch]

            frames_tensor = torch.tensor(frames_batch, dtype=torch.float32).to(device)  # (batch, T, F_total)
            labels_tensor = torch.tensor(labels_batch, dtype=torch.long).to(device)     # (batch,) class indices

            outputs = model(frames_tensor)  # (batch, num_classes)
            loss = criterion(outputs, labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc_history.append((outputs.argmax(dim=1) == labels_tensor).float().mean().item())

        with torch.no_grad():
            model.eval()
            val_frames = torch.tensor([item[0] for item in validation_data], dtype=torch.float32).to(device)
            val_labels = torch.tensor([item[2] for item in validation_data], dtype=torch.long).to(device)
            val_outputs = model(val_frames)
            val_acc = (val_outputs.argmax(dim=1) == val_labels).float().mean().item()
            val_acc_history.append(val_acc)

    results = {
        "train_acc": float(np.mean(train_acc_history)) if train_acc_history else 0.0,
        "val_acc": float(np.mean(val_acc_history)) if val_acc_history else 0.0
    }

    return model, results

def test_model(model, test_data: list) -> dict:
    # test_data is a list of (features, animal, label_idx) where features is (T, F_total) and label_idx is an int
    device = get_device()
    model.eval()
    with torch.no_grad():
        test_frames = torch.tensor([item[0] for item in test_data], dtype=torch.float32).to(device)
        test_labels = torch.tensor([item[2] for item in test_data], dtype=torch.long).to(device)
        test_outputs = model(test_frames)
        test_acc = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()

    return {"test_acc": test_acc}

def run():
    start = time()
    video_ids = get_video_ids()
    train_video_ids, validation_test_video_ids, test_video_ids = split_video_ids(video_ids)

    # preprocessing data and running limb-tracking inference
    train_video_as_frames = get_or_build_cached(
        "train_video_as_frames",
        lambda: [(get_video_frames(video_id), animal, behavior) for video_id, animal, behavior in train_video_ids],
    )
    validation_test_video_as_frames = get_or_build_cached(
        "validation_test_video_as_frames",
        lambda: [(get_video_frames(video_id), animal, behavior) for video_id, animal, behavior in validation_test_video_ids],
    )
    test_video_as_frames = get_or_build_cached(
        "test_video_as_frames",
        lambda: [(get_video_frames(video_id), animal, behavior) for video_id, animal, behavior in test_video_ids],
    )
    train_vidoes_as_limbs = get_or_build_cached(
        "train_videos_as_limbs",
        lambda: [(np.array(run_limb_tracking(frames, animal)), animal, behavior) for frames, animal, behavior in train_video_as_frames],
    )
    validation_test_videos_as_limbs = get_or_build_cached(
        "validation_test_videos_as_limbs",
        lambda: [(np.array(run_limb_tracking(frames, animal)), animal, behavior) for frames, animal, behavior in validation_test_video_as_frames],
    )
    test_videos_as_limbs = get_or_build_cached(
        "test_videos_as_limbs",
        lambda: [(np.array(run_limb_tracking(frames, animal)), animal, behavior) for frames, animal, behavior in test_video_as_frames],
    )

    print(f"train videos as limbs len: {len(train_vidoes_as_limbs)}, validation/test videos as limbs len: {len(validation_test_videos_as_limbs)}, test videos as limbs len: {len(test_videos_as_limbs)}")
    train_videos_normalized = [normalize_frames(frames, animal, behavior) for frames, animal, behavior in train_vidoes_as_limbs]
    validation_test_videos_normalized = [normalize_frames(frames, animal, behavior) for frames, animal, behavior in validation_test_videos_as_limbs]
    test_videos_normalized = [normalize_frames(frames, animal, behavior) for frames, animal, behavior in test_videos_as_limbs]

    train_videos_features = [flatten_frame_features(frames) for frames in train_videos_normalized]  # (T, L, F_aug) -> (T, F_total)
    validation_videos_features = [flatten_frame_features(frames) for frames in validation_test_videos_normalized]  # (T, L, F_aug) -> (T, F_total)
    test_videos_features = [flatten_frame_features(frames) for frames in test_videos_normalized]  # (T, L, F_aug) -> (T, F_total)

    # Build canonical supervised samples: (features, animal, behavior_class_index).
    train_data = [
        (frames, animal, BEHAVIORS.index(behavior))
        for frames, (_, animal, behavior) in zip(train_videos_features, train_vidoes_as_limbs)
    ]
    validation_data = [
        (frames, animal, BEHAVIORS.index(behavior))
        for frames, (_, animal, behavior) in zip(validation_videos_features, validation_test_videos_as_limbs)
    ]
    test_data = [
        (frames, animal, BEHAVIORS.index(behavior))
        for frames, (_, animal, behavior) in zip(test_videos_features, test_videos_as_limbs)
    ]

    print(f"Data preprocessing and limb tracking done at {time() - start:.2f} seconds")

    # train the model 
    model, train_results = train_model(train_data, validation_data)
    print(train_results)
    print(f"Model training done at {time() - start:.2f} seconds")

    # evaluate the model
    test_results = test_model(model, test_data)
    print(test_results)
    print(f"Model testing done at {time() - start:.2f} seconds")

    # save the model
    torch.save(model.state_dict(), "behavior_cnn.pth")
    print(f"Model saved at {time() - start:.2f} seconds")

if __name__ == "__main__":
    run()