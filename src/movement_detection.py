import math
import pandas as pd
import numpy as np
from pathlib import Path

from paths import RESULTS_DIR

SEGMENTS = {
    "upper" : ["wrist", "elbow", "shoulder"],
    "lower" : ["hip", "knee", "ankle"],
    "torso" : ["shoulder", "hip"],
}

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

def normalize_data(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Normalize the DataFrame by scaling the x and y coordinates based on the center of mass of the keypoints.

    Args:
        df (pd.DataFrame): The input DataFrame with keypoint coordinates.
        direction (str): The direction of movement ("left" or "right").
    
    Returns:
        pd.DataFrame: The normalized DataFrame with scaled coordinates.
    """

    # Calculate the center of mass for each frame
    df['center_x'] = df.apply(lambda row: center_of_mass({k: (row[f'x_{k}'], row[f'y_{k}']) for k in SEGMENTS["torso"]}), axis=1).apply(lambda x: x[0])
    df['center_y'] = df.apply(lambda row: center_of_mass({k: (row[f'x_{k}'], row[f'y_{k}']) for k in SEGMENTS["torso"]}), axis=1).apply(lambda x: x[1])

    # Normalize the coordinates by subtracting the center of mass
    for keypoint in df.columns[df.columns.str.startswith('x_')].str.replace('x_', ''):
        df[f'x_{keypoint}'] = df[f'x_{keypoint}'] - df['center_x']
        df[f'y_{keypoint}'] = df[f'y_{keypoint}'] - df['center_y']

    # Invert the x-coordinates for left direction
    if direction == "left":
        for keypoint in df.columns[df.columns.str.startswith('x_')].str.replace('x_', ''):
            df[f'x_{keypoint}'] = -df[f'x_{keypoint}']

    # Drop the center of mass columns
    df.drop(columns=['center_x', 'center_y'], inplace=True)

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

    # save_csv(pivoted_df, FOLDER / "baby_Cut_Crawli1_BB006_0_blur_pivoted.csv")