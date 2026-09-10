import math
import pandas as pd
import numpy as np
from pathlib import Path

from src.paths import RESULTS_DIR

SEGMENTS = {
    "upper" : ["wrist", "elbow", "shoulder"],
    "lower" : ["hip", "knee", "ankle"],
    "torso" : ["shoulder", "hip"],
}

RIGID_PAIRS = [
    ("shoulder", "hip"),      # tronc
    ("hip", "knee"),          # fémur
    ("knee", "ankle"),        # tibia
    ("shoulder", "elbow"),    # humérus
    ("elbow", "wrist"),       # avant-bras
]

def load_data(file_path: str, direction: str, confidence_threshold: float = 0.5) -> pd.DataFrame:
    """
    Load the CSV data into a pivoted pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        confidence_threshold (float): The minimum confidence level for a keypoint to be considered valid.
                                      (default: 0.5)
        direction (str): The direction of movement ("left" or "right").

    Returns:
        pd.DataFrame: The loaded and pivoted DataFrame.
    """
    csv_path = Path(file_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {
        "frame",
        "person_id",
        "keypoint",
        "confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Colonnes manquantes dans le CSV : "
            + ", ".join(sorted(missing))
        )

    # Determine which x/y columns to use
    if {"x_clean", "y_clean"}.issubset(df.columns):
        x_col = "x_clean"
        y_col = "y_clean"
    elif {"x", "y"}.issubset(df.columns):
        x_col = "x"
        y_col = "y"
    else:
        raise ValueError(
            "Le CSV doit contenir x/y ou x_clean/y_clean."
        )

    df = df.copy()

    # Convert relevant columns to numeric, coercing errors to NaN
    for col in ["frame", "person_id", "confidence", x_col, y_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out rows with low confidence
    df = df[df["confidence"] >= confidence_threshold]

    # Refactor and normalize the DataFrame
    df = pivot_data(df)
    df = normalize_data(df, direction)

    return df

def pivot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the DataFrame to have keypoints as columns and their coordinates as values.

    Args:
        df (pd.DataFrame): The input DataFrame with 'keypoint', 'x', and 'y' columns.

    Returns:
        pd.DataFrame: The pivoted DataFrame (frame / x_keypoint / ... / y_keypoint).
    """

    # Pivot the DataFrame to have keypoints as columns and their coordinates as values
    pivoted_df = df.pivot(index='frame', columns='keypoint', values=['x', 'y'])

    # Flatten the MultiIndex columns
    pivoted_df.columns = [f"{coord}_{normalize_keypoint(keypoint)}" for coord, keypoint in pivoted_df.columns]

    return pivoted_df

def normalize_data(df: pd.DataFrame, direction: str,
                   alpha: float = 0.1) -> pd.DataFrame:
    # 1. Centre de masse (ton code actuel, simplifié)
    torso = SEGMENTS["torso"]
    df['center_x'] = df[[f'x_{k}' for k in torso]].mean(axis=1)
    df['center_y'] = df[[f'y_{k}' for k in torso]].mean(axis=1)

    # 2. Translation
    kp_cols = [c for c in df.columns if c.startswith(('x_', 'y_'))]
    for kp in [c[2:] for c in kp_cols if c.startswith('x_')]:
        df[f'x_{kp}'] -= df['center_x']
        df[f'y_{kp}'] -= df['center_y']

    # 3. Scale brut + lissage
    scale_raw = df.apply(compute_scale_raw, axis=1)
    df['scale'] = smooth_scale(scale_raw, alpha=alpha)

    # 4. Division par le scale
    for kp in [c[2:] for c in kp_cols if c.startswith('x_')]:
        df[f'x_{kp}'] /= df['scale']
        df[f'y_{kp}'] /= df['scale']

    # 5. Miroir pour "left"
    if direction == "left":
        for kp in [c[2:] for c in kp_cols if c.startswith('x_')]:
            df[f'x_{kp}'] = -df[f'x_{kp}']

    # 6. Calcul des angles corporels
    df = calculate_body_angles(df)

    # 7. Nettoyage
    df.drop(columns=['center_x', 'center_y'], inplace=True)
    # optionnel : garder 'scale' pour debug, sinon df.drop(columns=['scale'], inplace=True)

    return df
def normalize_keypoint(keypoint: str) -> str:
    """
    Normalize a keypoint string by removing its side designation (e.g., 'left', 'right')
    and converting it to lowercase.

    Args:
        keypoint (str): The keypoint string to normalize.

    Returns:
        str: The normalized keypoint string.
    """
    return keypoint.split(' ')[-1].lower()

def center_of_mass(keypoints: dict) -> tuple:
    """
    Calculate the center of mass of a set of keypoints.

    Args:
        keypoints (dict): A dictionary of keypoints with their coordinates.

    Returns:
        tuple: The (x, y) coordinates of the center of mass.
    """
    x_coords = [coord[0] for coord in keypoints.values()]
    y_coords = [coord[1] for coord in keypoints.values()]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    return (center_x, center_y)

def compute_scale_raw(row: pd.Series) -> float:
    """
    Calcule une longueur corporelle de référence pour une frame,
    comme la médiane des longueurs de segments rigides disponibles.
    """
    lengths = []
    for a, b in RIGID_PAIRS:
        xa, ya = row.get(f"x_{a}"), row.get(f"y_{a}")
        xb, yb = row.get(f"x_{b}"), row.get(f"y_{b}")
        if pd.notna(xa) and pd.notna(ya) and pd.notna(xb) and pd.notna(yb):
            lengths.append(math.hypot(xa - xb, ya - yb))
    if not lengths:
        return np.nan
    return float(np.median(lengths))

def smooth_scale(scale_raw: pd.Series, alpha: float = 0.1,
                 outlier_ratio: float = 0.2) -> pd.Series:
    """
    Lisse la série de scale par EMA, en ignorant les outliers.
    - alpha : poids de la nouvelle mesure (petit = stable, lent)
    - outlier_ratio : écart relatif toléré avant de rejeter L_t
    """
    smoothed = np.full(len(scale_raw), np.nan)
    prev = np.nan

    for i, L in enumerate(scale_raw.values):
        if np.isnan(L):
            smoothed[i] = prev  # garde la dernière valeur valide
            continue

        if np.isnan(prev):
            # initialisation : 1ère mesure valide
            prev = L
        elif abs(L - prev) / prev > outlier_ratio:
            # outlier probable → on ignore L, on garde prev
            pass
        else:
            prev = alpha * L + (1 - alpha) * prev

        smoothed[i] = prev

    return pd.Series(smoothed, index=scale_raw.index)

def angle_from_points(A: tuple, B: tuple, C: tuple) -> float:
    """
    Calculate the angle at point B formed by points A, B, and C.

    Args:
        A (tuple): Coordinates of point A (x_A, y_A).
        B (tuple): Coordinates of point B (x_B, y_B).
        C (tuple): Coordinates of point C (x_C, y_C).

    Returns:
        float: The angle at point B in degrees.
    """
    # Vectors BA and BC
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # Calculate the dot product and magnitudes
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    magnitude_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

    if magnitude_BA == 0 or magnitude_BC == 0:
        return np.nan  # Avoid division by zero

    # Calculate the angle in radians and convert to degrees
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    angle_rad = math.acos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def calculate_body_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the angles of body segments based on keypoint coordinates.

    Args:
        df (pd.DataFrame): The input DataFrame with keypoint coordinates.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for body segment angles.
    """
    a, b, c = SEGMENTS["upper"]
    df[f"angle_upper"] = df.apply(
        lambda row: angle_from_points(
            (row[f"x_{a}"], row[f"y_{a}"]),
            (row[f"x_{b}"], row[f"y_{b}"]),
            (row[f"x_{c}"], row[f"y_{c}"])
            ),
            axis=1
        )

    a, b, c = SEGMENTS["lower"]
    df[f"angle_lower"] = df.apply(
        lambda row: angle_from_points(
            (row[f"x_{a}"], row[f"y_{a}"]),
            (row[f"x_{b}"], row[f"y_{b}"]),
            (row[f"x_{c}"], row[f"y_{c}"])
            ),
            axis=1
        )

    return df

if __name__ == "__main__":

    FOLDER = RESULTS_DIR / "Cut_Crawli1_BB006_0_blur"
    csv_path = FOLDER / "baby_Cut_Crawli1_BB006_0_blur_clean.csv"

    # Test pivot_data function
    df = load_data(csv_path, direction="left", confidence_threshold=0.5)

    print(df.head())

    def save_csv(df: pd.DataFrame, output_path: Path):
        """
        Save the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            output_path (Path): The path where the CSV file will be saved.
        """
        df.to_csv(output_path)

    save_csv(df, FOLDER / "baby_Cut_Crawli1_BB006_0_blur_normalized.csv")