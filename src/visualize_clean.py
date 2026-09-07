from pathlib import Path

import cv2
import pandas as pd

from src.paths import (
    VIDEOS_DIR,
    RESULTS_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

# Nom du fichier vidéo
VIDEO_EXTENSION = ".mp4"

# Afficher uniquement les points suffisamment fiables
CONF_THRESHOLD = 0.5

# Rayon des points
POINT_RADIUS = 5

# Épaisseur des lignes du squelette
LINE_THICKNESS = 2


# ============================================================
# SQUELETTE COCO
# ============================================================

SKELETON = [
    ("Nose", "Left Eye"),
    ("Nose", "Right Eye"),
    ("Left Eye", "Left Ear"),
    ("Right Eye", "Right Ear"),

    ("Left Shoulder", "Right Shoulder"),

    ("Left Shoulder", "Left Elbow"),
    ("Left Elbow", "Left Wrist"),

    ("Right Shoulder", "Right Elbow"),
    ("Right Elbow", "Right Wrist"),

    ("Left Shoulder", "Left Hip"),
    ("Right Shoulder", "Right Hip"),

    ("Left Hip", "Right Hip"),

    ("Left Hip", "Left Knee"),
    ("Left Knee", "Left Ankle"),

    ("Right Hip", "Right Knee"),
    ("Right Knee", "Right Ankle"),
]


# ============================================================
# OUTILS
# ============================================================

def draw_keypoints(frame, person_df):
    """
    Dessine les keypoints nettoyés et le squelette.
    """

    points = {}

    for _, row in person_df.iterrows():

        x = row["x_clean"]
        y = row["y_clean"]

        if pd.isna(x) or pd.isna(y):
            continue

        if row["confidence"] < CONF_THRESHOLD:
            continue

        keypoint = row["keypoint"]

        points[keypoint] = (
            int(round(x)),
            int(round(y))
        )

    # --------------------------------------------------------
    # Squelette
    # --------------------------------------------------------

    for kp1, kp2 in SKELETON:

        if kp1 not in points or kp2 not in points:
            continue

        p1 = points[kp1]
        p2 = points[kp2]

        cv2.line(
            frame,
            p1,
            p2,
            (0, 255, 0),
            LINE_THICKNESS
        )

    # --------------------------------------------------------
    # Keypoints
    # --------------------------------------------------------

    for keypoint, (x, y) in points.items():

        cv2.circle(
            frame,
            (x, y),
            POINT_RADIUS,
            (0, 0, 255),
            -1
        )

    return frame


# ============================================================
# VISUALISATION
# ============================================================

def create_visualization(input_folder, progress_callback=None):

    if progress_callback is not None:
        progress_callback(0)

    folder = RESULTS_DIR / input_folder

    csv_path = (
        folder /
        f"baby_{input_folder}_clean.csv"
    )

    video_path = (
        VIDEOS_DIR /
        f"{input_folder}{VIDEO_EXTENSION}"
    )

    output_path = (
        folder /
        f"visualization_{input_folder}.mp4"
    )

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Vidéo
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video:\n{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # --------------------------------------------------------
    # VideoWriter
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # Traitement frame par frame
    # --------------------------------------------------------

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Keypoints de cette frame
        # ----------------------------------------------------

        frame_df = df[
            df["frame"] == frame_number
        ]

        # ----------------------------------------------------
        # Plusieurs personnes possibles
        # ----------------------------------------------------

        for person_id, person_df in frame_df.groupby(
            "person_id"
        ):

            frame = draw_keypoints(
                frame,
                person_df
            )

            # Affichage de l'ID
            valid_points = person_df[
                person_df["x_clean"].notna()
                & person_df["y_clean"].notna()
            ]

            if not valid_points.empty:

                x = int(
                    valid_points["x_clean"].mean()
                )

                y = int(
                    valid_points["y_clean"].mean()
                )

                cv2.putText(
                    frame,
                    f"ID: {int(person_id)}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

        # ----------------------------------------------------
        # Texte frame
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        frame_number += 1

        if frame_number % 10 == 0 or frame_number == total_frames:
            if progress_callback is not None:
                progress_callback(
                    frame_number / total_frames * 100
                )

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print(
        f"Visualization saved to:\n{output_path}"
    )

    return output_path

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    FOLDER = "Cut_Crawli1_BB006_0_blur_cut"
    
    # Generate clean data if csv not found
    clean_csv_path = (
        RESULTS_DIR /
        FOLDER /
        f"baby_{FOLDER}_clean.csv"
    )

    if not clean_csv_path.exists():
        from src.clean_data import clean_csv

        input_csv_path = (
            RESULTS_DIR /
            FOLDER /
            f"baby_{FOLDER}.csv"
        )

        print(f"Cleaning data: {input_csv_path}")
        clean_csv(input_csv_path)

    print(f"Visualizing: {clean_csv_path}")
    video_path = create_visualization(FOLDER)

    # Show the video
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
