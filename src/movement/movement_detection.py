import pandas as pd
import numpy as np

from src.movement.normalize_data import load_data

from src.paths import RESULTS_DIR

from scipy.signal import savgol_filter

def detect_movements(angles, 
                     fps=30, 
                     threshold=30.0,
                     max_window_sec=4.0):
    """
    Détecte les mouvements en cherchant les fenêtres où l'angle
    a varié de plus de `threshold` degrés.
    """
    a = angles.interpolate(limit_direction='both').values
    n = len(a)
    max_window = int(max_window_sec * fps)

    # Pour chaque frame de départ, chercher la première frame
    # où |a[i] - a[j]| >= threshold
    events = []
    i = 0
    while i < n:
        found = False
        for w in range(1, max_window):
            if i + w >= n:
                break
            delta = a[i + w] - a[i]
            if abs(delta) >= threshold:
                # Chercher l'extrême dans cette direction
                sens = np.sign(delta)
                j = i + w
                a_extreme = a[j]
                while j + 1 < n and (a[j + 1] - a_extreme) * sens > 0:
                    a_extreme = a[j + 1]
                    j += 1
                amplitude = a_extreme - a[i]
                events.append({
                    "frame_start": angles.index[i],
                    "frame_end": angles.index[j],
                    "angle_start": a[i],
                    "angle_end": a_extreme,
                    "amplitude": amplitude,
                    "type": "flexion" if amplitude < 0 else "extension",
                    "duration": j - i,
                })
                i = j
                found = True
                break
        if not found:
            i += 1

    return pd.DataFrame(events)

def show_detected_movements(angles: pd.Series, movements: pd.DataFrame):
    """
    Affiche un plot des angles avec les mouvements détectés.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.figure(figsize=(12, 6))
    plt.plot(angles.index, angles, label='Angle Upper', color='blue')

    colors = {'flexion': 'red', 'extension': 'green'}

    for _, row in movements.iterrows():
        color = colors.get(row['type'], 'gray')
        plt.axvspan(row['frame_start'], row['frame_end'], color=color, alpha=0.3)
        plt.text((row['frame_start'] + row['frame_end']) / 2,
                max(angles.dropna()),
                row['type'], ha='center', va='bottom', fontsize=8)

    # Légende manuelle pour les types
    legend_elements = [
        plt.Line2D([0], [0], color='blue', label='Angle Upper'),
        Patch(facecolor='red', alpha=0.3, label='Flexion'),
        Patch(facecolor='green', alpha=0.3, label='Extension'),
    ]
    plt.legend(handles=legend_elements)

    plt.title('Angle Upper with Detected Movements')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.grid()
    plt.show()

if __name__ == "__main__":

    FOLDER = RESULTS_DIR / "Cut_Crawli1_BB006_0_blur"
    csv_path = FOLDER / "baby_Cut_Crawli1_BB006_0_blur_clean.csv"

    # Test pivot_data function
    df = load_data(csv_path, direction="left", confidence_threshold=0.5)

    print(df.head())

    print("\nDetecting movements...")
    movements = detect_movements(df['angle_upper'], 
                                 fps=30, 
                                 threshold=30.0,
                                 max_window_sec=2.0)

    # Affiche les mouvements détectés
    print(movements)

    # Affiche le plot des mouvements détectés
    show_detected_movements(df['angle_upper'], movements)