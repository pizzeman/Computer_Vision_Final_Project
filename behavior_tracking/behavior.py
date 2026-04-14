import pandas as pd
import ast
import cv2

METADATA_FILENAME = "AR_metadata.xlsx"
ANIMALS = ["horse"]
BEHAVIORS = ["running"]
TEST_RATIO = 0.2
SAMPLES_PER_SEC = 10

def sanitize(s: str) -> str:
    return s.strip().lower().replace(",", "").replace("[","").replace("]","").replace("'","").replace('"',"")

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
        if sanitize(animal_action[1]) not in BEHAVIORS:
            continue
        video_ids.append((row.video_id, animal, sanitize(animal_action[1])))
    return video_ids

def split_video_ids(video_ids: list) -> tuple:
    n_test = int(len(video_ids) * TEST_RATIO)
    test_video_ids = video_ids[:n_test]
    train_video_ids = video_ids[n_test:]
    return train_video_ids, test_video_ids

def split_video_into_frames(video_id: str, i: int) -> list:
    filename = f"{video_id}_{i:6d}.mp4"
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        raise Exception(f"Could not open video file: {filename}")
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % int(fps / SAMPLES_PER_SEC) == 0:
            frames.append(frame)
        frame_count += 1
    video.release()
    return frames