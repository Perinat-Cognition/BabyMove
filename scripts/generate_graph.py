"""
Génère un graphe simple des trajectoires X et Y dans le temps
(sans les données brutes), à partir de load_data.

Le script utilise `load_data` depuis `src.movement_detection` pour charger
les données. Les colonnes renvoyées sont de la forme :
    frame, x_<keypoint>, y_<keypoint>, ...
où les valeurs sont déjà lissées et où le seuil de confiance a déjà été
appliqué (valeurs NaN là où confidence < threshold).

Exemples:
    python generate_graph.py --input results/baby/baby_clean.csv --keypoint ankle
    python generate_graph.py --input results/baby/baby_clean.csv --all-keypoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.movement_detection import load_data


def get_keypoints_from_df(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des keypoints à partir d'un DataFrame pivoté.

    On considère qu'un keypoint est présent s'il possède une colonne
    'x_<keypoint>' ET une colonne 'y_<keypoint>'.
    """
    keypoints = []
    for col in df.columns:
        if col.startswith("x_"):
            kp = col[2:]
            if f"y_{kp}" in df.columns:
                keypoints.append(kp)
    return keypoints


def ask_from_list(options: list[str]) -> str:
    """Demande à l'utilisateur de choisir un élément dans une liste."""
    print("\nChoisissez un élément :")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        choice = input("Entrez le numéro correspondant : ")
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(options):
                return options[index]
        print("Choix invalide. Veuillez réessayer.")


def plot_smooth_with_gaps(
    ax,
    time,
    values,
    color,
    label,
    max_gap=2,
):
    """Trace les données lissées en interrompant le tracé sur les lacunes."""
    t = np.asarray(time)
    v = np.asarray(values)

    if len(t) != len(v):
        raise ValueError(
            f"time et values doivent avoir la même longueur : "
            f"time={len(t)}, values={len(v)}"
        )

    valid_mask = ~np.isnan(v)
    if not np.any(valid_mask):
        return

    valid_indices = np.where(valid_mask)[0]
    gaps = np.diff(valid_indices) - 1
    gap_indices = np.where(gaps > max_gap)[0]

    if len(gap_indices) == 0:
        ax.plot(
            t[valid_indices],
            v[valid_indices],
            color=color,
            linewidth=1.5,
            label=label,
        )
        return

    start_idx = 0
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        segment_indices = valid_indices[start_idx:end_idx]
        if len(segment_indices) > 0:
            ax.plot(
                t[segment_indices],
                v[segment_indices],
                color=color,
                linewidth=1.5,
            )
        start_idx = end_idx

    if start_idx < len(valid_indices):
        segment_indices = valid_indices[start_idx:]
        ax.plot(
            t[segment_indices],
            v[segment_indices],
            color=color,
            linewidth=1.5,
        )

    ax.plot([], [], color=color, linewidth=1.5, label=label)


def build_plot(
    input_csv: Path,
    keypoint: str,
    direction: str = "left",
    confidence_threshold: float = 0.5,
    output_png: Path | None = None,
) -> None:
    """Charge le CSV via load_data et génère le graphique (sans brut)."""

    # ------------------------------------------------------------------
    # Chargement via load_data
    # ------------------------------------------------------------------
    df = load_data(
        input_csv,
        direction=direction,
        confidence_threshold=confidence_threshold,
    )

    # ------------------------------------------------------------------
    # Vérifier la présence du keypoint
    # ------------------------------------------------------------------
    x_col = f"x_{keypoint}"
    y_col = f"y_{keypoint}"

    if x_col not in df.columns or y_col not in df.columns:
        available = get_keypoints_from_df(df)
        raise ValueError(
            f"Keypoint '{keypoint}' non trouvé dans le DataFrame.\n"
            f"Keypoints disponibles : {available}"
        )

    # ------------------------------------------------------------------
    # Axe temporel
    # ------------------------------------------------------------------
    time = df["frame"] if "frame" in df.columns else df.index

    # ------------------------------------------------------------------
    # Données (déjà lissées et filtrées par load_data)
    # ------------------------------------------------------------------
    x_smooth = df[x_col]
    y_smooth = df[y_col]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # X
    plot_smooth_with_gaps(
        axes[0],
        time,
        x_smooth,
        color="#08519c",
        label=f"x {keypoint} (lissé)",
    )
    axes[0].set_ylabel("x")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    # Y
    plot_smooth_with_gaps(
        axes[1],
        time,
        y_smooth,
        color="#cb181d",
        label=f"y {keypoint} (lissé)",
    )
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("frame")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Trajectoires X et Y de {keypoint} en fonction du temps",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    output_dir = input_csv.parent / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_png is None:
        safe_kp = keypoint.replace(" ", "_").lower()
        output_png = output_dir / f"{safe_kp}_trajectories.png"

    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    print(f"Graphique sauvegardé : {output_png}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère un graphe des trajectoires x/y (sans brut)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Chemin vers le CSV.")
    parser.add_argument("--keypoint", help="Nom du keypoint (ex: 'ankle', 'knee').")
    parser.add_argument(
        "--direction",
        default="left",
        choices=["left", "right"],
        help="Direction du corps à charger.",
    )
    parser.add_argument("--output", type=Path, help="Chemin de sortie PNG.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Seuil de confiance (entre 0 et 1).",
    )
    parser.add_argument(
        "--all-keypoints",
        action="store_true",
        help="Génère un graphique pour chaque keypoint.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Fichier CSV introuvable : {args.input}")

    # Charger une fois pour lister les keypoints
    df_probe = load_data(
        args.input,
        direction=args.direction,
        confidence_threshold=args.confidence,
    )
    keypoints = get_keypoints_from_df(df_probe)

    if not keypoints:
        parser.error(f"Aucun keypoint trouvé dans : {args.input}")

    if args.all_keypoints:
        for kp in keypoints:
            print(f"\nGénération du graphique pour le keypoint : {kp}")
            build_plot(
                args.input,
                kp,
                args.direction,
                args.confidence,
                args.output,
            )
    else:
        if not args.keypoint:
            args.keypoint = ask_from_list(keypoints)

        build_plot(
            args.input,
            args.keypoint,
            args.direction,
            args.confidence,
            args.output,
        )


if __name__ == "__main__":
    main()