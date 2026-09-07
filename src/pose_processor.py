from pathlib import Path
import csv
import torch
from ultralytics import YOLO
import cv2

LEFT_SIDE_KEYPOINTS = {
    0: "Nose",
    1: "Left Eye",
    3: "Left Ear",
    5: "Left Shoulder",
    7: "Left Elbow",
    9: "Left Wrist",
    11: "Left Hip",
    13: "Left Knee",
    15: "Left Ankle",
}

RIGHT_SIDE_KEYPOINTS = {
    0: "Nose",
    2: "Right Eye",
    4: "Right Ear",
    6: "Right Shoulder",
    8: "Right Elbow",
    10: "Right Wrist",
    12: "Right Hip",
    14: "Right Knee",
    16: "Right Ankle",
}

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "yolo26x-pose.pt"
RESULTS_DIR = PROJECT_DIR / "results"
VIDEOS_DIR = PROJECT_DIR / "videos"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

def process_video(
    video_path,
    direction,
    results_dir=RESULTS_DIR,
    model_path=MODEL_PATH,
    progress_callback=None
):
    video_path = Path(video_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    selected_keypoints = LEFT_SIDE_KEYPOINTS if direction == "left" else RIGHT_SIDE_KEYPOINTS

    video_name = video_path.stem

    # Nombre total de frames
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model = YOLO(model_path)
    model.to(DEVICE)
    
    output_dir = results_dir / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir = results_dir / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{video_name}.csv"

    # stream=True = résultats fournis frame par frame
    results = model.track(
        str(video_path),
        persist=True,
        save=True,
        verbose=False,
        stream=True,
        project=str(results_dir),
        name=video_name,
        exist_ok=True
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "person_id",
            "keypoint",
            "x",
            "y",
            "confidence"
        ])

        for frame, result in enumerate(results):

            if result.keypoints is not None:

                keypoints = result.keypoints.xy.cpu().numpy()

                confidences = result.keypoints.conf

                if confidences is not None:
                    confidences = confidences.cpu().numpy()

                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                else:
                    ids = [-1] * len(keypoints)

                for person_idx, person_keypoints in enumerate(keypoints):

                    person_id = ids[person_idx]

                    for kp_idx, (x, y) in enumerate(person_keypoints):

                        conf = (
                            confidences[person_idx][kp_idx]
                            if confidences is not None
                            else None
                        )
                        if kp_idx in selected_keypoints:
                            writer.writerow([
                                frame,
                                int(person_id),
                                selected_keypoints.get(kp_idx, "Unknown"),
                                float(x),
                                float(y),
                                float(conf) if conf is not None else None
                            ])

            # Mise à jour de la progression
            if progress_callback and total_frames > 0:

                progress = ((frame + 1) / total_frames) * 100

                progress_callback(progress)

    return output_csv