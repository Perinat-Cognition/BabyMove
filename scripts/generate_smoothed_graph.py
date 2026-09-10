"""
Génère un graphe simple des trajectoires X et Y dans le temps.

Le script lit `results/[nom]/baby_[nom]_clean.csv` et affiche deux sous-graphes
superposant les données brutes et lissées.

- Les données brutes sont affichées avec une transparence dépendant de la confiance.
- Les données lissées ne sont affichées que lorsque confidence >= 0.5.
- Les zones où les données sont absentes ou de faible confiance sont laissées
  vides afin de visualiser les interruptions dans la trajectoire.

Exemples:
    python generate_smoothed_graph.py --name mon_bebe --keypoint "Left Hip"
    python generate_smoothed_graph.py \
        --input results/mon_bebe/baby_mon_bebe_clean.csv \
        --keypoint "Left Hip"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _default_input(name: str) -> Path:
    """Construit le chemin par défaut vers le fichier CSV."""
    return Path("results") / name / f"baby_{name}_clean.csv"

def get_results_names() -> list[str]:
    """Retourne la liste des noms de dossiers dans results/."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []
    return [
        d.name for d in results_dir.iterdir()
        if d.is_dir() and (d / f"baby_{d.name}_clean.csv").exists()
    ]

def get_keypoints_from_csv(csv_path: Path) -> list[str]:
    """Retourne la liste des keypoints présents dans le CSV."""
    df = pd.read_csv(csv_path)
    if "keypoint" not in df.columns:
        raise ValueError(
            f"La colonne 'keypoint' est absente du CSV : {csv_path}"
        )
    return df["keypoint"].dropna().unique().tolist()

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

def plot_raw_with_confidence(
    ax,
    time,
    values,
    confidence,
    label,
    color,
):
    """
    Trace les données brutes avec une transparence dépendant de la confiance.

    Les points avec une confiance < 0.5 sont affichés avec une faible
    transparence, tandis que les points avec une confiance >= 0.5 sont
    plus visibles.
    """

    # Courbe brute légère pour conserver la continuité visuelle
    ax.plot(
        time,
        values,
        color=color,
        alpha=0.75,
        linewidth=0.8,
    )

    # Transparence selon la confiance
    alpha_values = confidence.apply(
        lambda c: 0.15 if c < 0.5 else 0.85
    )

    # Points individuels
    for t, value, alpha in zip(time, values, alpha_values):
        if pd.notna(value):
            ax.scatter(
                t,
                value,
                color=color,
                alpha=alpha,
                s=12,
            )

    # Entrée de légende
    ax.scatter(
        [],
        [],
        color=color,
        alpha=0.85,
        s=12,
        label=label,
    )


def plot_smooth_with_gaps(
    ax,
    time,
    values,
    color,
    label,
    max_gap=2,
):
    """
    Trace les données lissées en interrompant le tracé lorsqu'il y a
    des lacunes.

    Args:
        ax: axe matplotlib sur lequel tracer.
        time: valeurs temporelles (frames).
        values: valeurs X ou Y lissées.
        color: couleur de la courbe.
        label: nom affiché dans la légende.
        max_gap: nombre maximum de frames consécutives manquantes autorisées
                 avant de couper le tracé.
    """

    # Conversion en numpy arrays
    t = np.asarray(time)
    v = np.asarray(values)

    # Vérification de sécurité
    if len(t) != len(v):
        raise ValueError(
            f"time et values doivent avoir la même longueur : "
            f"time={len(t)}, values={len(v)}"
        )

    # Identifier les valeurs valides
    valid_mask = ~np.isnan(v)

    # Aucune donnée valide
    if not np.any(valid_mask):
        return

    # Indices correspondant aux valeurs valides
    valid_indices = np.where(valid_mask)[0]

    # Détecter les lacunes importantes.
    #
    # Exemple :
    # indices valides = [0, 1, 2, 5, 6]
    #
    # Entre 2 et 5, il manque les indices 3 et 4.
    #
    # Si max_gap = 2, cette lacune est encore autorisée.
    #
    # On coupe uniquement lorsque le nombre de frames manquantes
    # est supérieur à max_gap.
    gaps = np.diff(valid_indices) - 1

    gap_indices = np.where(gaps > max_gap)[0]

    # Aucun gros trou : on trace toute la courbe
    if len(gap_indices) == 0:
        ax.plot(
            t[valid_indices],
            v[valid_indices],
            color=color,
            linewidth=1.5,
            label=label,
        )
        return

    # Tracer les différents segments
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

    # Tracer le dernier segment
    if start_idx < len(valid_indices):
        segment_indices = valid_indices[start_idx:]

        ax.plot(
            t[segment_indices],
            v[segment_indices],
            color=color,
            linewidth=1.5,
        )

    # Une seule entrée dans la légende
    ax.plot(
        [],
        [],
        color=color,
        linewidth=1.5,
        label=label,
    )


def build_plot(
    input_csv: Path,
    keypoint: str,
    confidence_threshold: float = 0.5,
    output_png: Path | None = None,
) -> None:
    """Charge le CSV et génère le graphique."""

    # ------------------------------------------------------------------
    # Lecture du CSV
    # ------------------------------------------------------------------

    df = pd.read_csv(input_csv)

    # Vérification de la présence de la colonne keypoint
    if "keypoint" not in df.columns:
        raise ValueError(
            "La colonne 'keypoint' est absente du CSV."
        )

    # Vérification du keypoint demandé
    available_keypoints = df["keypoint"].dropna().unique()

    if keypoint not in available_keypoints:
        raise ValueError(
            f"Keypoint '{keypoint}' non trouvé dans le CSV.\n"
            f"Keypoints disponibles : {list(available_keypoints)}"
        )

    # ------------------------------------------------------------------
    # Filtrer le keypoint
    # ------------------------------------------------------------------

    df = df[df["keypoint"] == keypoint].copy()

    # Réinitialiser l'index.
    #
    # Cela évite d'avoir des index provenant du DataFrame original
    # après le filtrage du keypoint.
    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Données brutes
    # ------------------------------------------------------------------

    time = df["frame"]

    x_raw = df["x"]
    y_raw = df["y"]

    confidence = df["confidence"]

    # ------------------------------------------------------------------
    # Données lissées
    # ------------------------------------------------------------------

    confidence_mask = confidence >= confidence_threshold

    x_smooth = df["x_clean"].where(confidence_mask)
    y_smooth = df["y_clean"].where(confidence_mask)

    # ------------------------------------------------------------------
    # Création de la figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
    )

    # ==================================================================
    # X
    # ==================================================================

    plot_raw_with_confidence(
        axes[0],
        time,
        x_raw,
        confidence,
        "x brut",
        "#9ecae1",
    )

    plot_smooth_with_gaps(
        axes[0],
        time,
        x_smooth,
        color="#08519c",
        label="x lissé",
    )

    axes[0].set_ylabel("x")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    # ==================================================================
    # Y
    # ==================================================================

    plot_raw_with_confidence(
        axes[1],
        time,
        y_raw,
        confidence,
        "y brut",
        "#fcbba1",
    )

    plot_smooth_with_gaps(
        axes[1],
        time,
        y_smooth,
        color="#cb181d",
        label="y lissé",
    )

    axes[1].set_ylabel("y")
    axes[1].set_xlabel("frame")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Titre
    # ------------------------------------------------------------------

    fig.suptitle(
        f"Trajectoires X et Y de {keypoint} en fonction du temps",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    
    output_dir = input_csv.parent / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_png is None:
        output_png = output_dir / f"{keypoint.replace(' ', '_').lower()}_trajectories.png"

    fig.savefig(
        output_png,
        dpi=160,
        bbox_inches="tight",
    )

    print(f"Graphique sauvegardé : {output_png}")

    # Affichage
    plt.show()

def build_scale_plot(
    input_csv: Path,
    output_png: Path | None = None,
) -> None:
    """Charge le CSV et génère le graphique de l'échelle."""

    # ------------------------------------------------------------------
    # Lecture du CSV
    # ------------------------------------------------------------------

    df = pd.read_csv(input_csv)

    # ------------------------------------------------------------------
    # Données brutes
    # ------------------------------------------------------------------

    time = df["frame"]
    scale = df["scale"]

    # ------------------------------------------------------------------
    # Création de la figure
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(11, 4),
    )

    # Affichage direct du graphique (pas de raw)
    ax.plot(
        time,
        scale,
        color="#08519c",
        linewidth=1.5,
        label="scale",
    )

    ax.set_ylabel("scale")
    ax.set_xlabel("frame")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Titre
    # ------------------------------------------------------------------

    fig.suptitle(
        "Échelle (scale) en fonction du temps",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    output_dir = input_csv.parent / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_png is None:
        output_png = output_dir / "scale_trajectories.png"

    fig.savefig(
        output_png,
        dpi=160,
        bbox_inches="tight",
    )

    print(f"Graphique sauvegardé : {output_png}")

    # Affichage
    plt.show()

def main() -> None:
    """Point d'entrée du programme."""

    parser = argparse.ArgumentParser(
        description=(
            "Génère un graphe lisible des trajectoires x/y."
        )
    )

    parser.add_argument(
        "--name",
        help="Nom du dossier sous results/[nom]/.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Chemin direct vers le CSV.",
    )

    parser.add_argument(
        "--keypoint",
        help="Nom du keypoint à filtrer (ex: 'Left Hip').",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Chemin de sortie PNG.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Seuil de confiance pour les données lissées (entre 0 et 1).",
    )

    parser.add_argument(
        "--all-keypoints",
        action="store_true",
        help="Génère un graphique pour chaque keypoint.",
    )

    parser.add_argument(
        "--scale",
        action="store_true",
        help="Génère un graphique pour l'échelle (scale).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Déterminer le fichier d'entrée
    # ------------------------------------------------------------------

    if args.name is None:
        results = get_results_names()
        if not results:
            parser.error(
                "Aucun dossier dans results/ contenant un CSV."
            )
        args.name = ask_from_list(results)

    input_csv = (
        args.input
        if args.input is not None
        else _default_input(args.name)
    )

    # Vérifier que le fichier existe
    if not input_csv.exists():
        parser.error(
            f"Fichier CSV introuvable : {input_csv}"
        )

    if args.scale:
        build_scale_plot(
            input_csv,
            output_png=args.output
        )
        return

    if not args.keypoint:
        keypoints = get_keypoints_from_csv(input_csv)
        if not keypoints:
            parser.error(
                f"Aucun keypoint trouvé dans le CSV : {input_csv}"
            )
        args.keypoint = ask_from_list(keypoints)

    # ------------------------------------------------------------------
    # Générer le graphique
    # ------------------------------------------------------------------

    if args.all_keypoints:
        keypoints = get_keypoints_from_csv(input_csv)
        for kp in keypoints:
            print(f"\nGénération du graphique pour le keypoint : {kp}")
            build_plot(
                input_csv,
                kp,
                args.confidence,
                args.output,
            )
    else:
        build_plot(
            input_csv,
            args.keypoint,
            args.confidence,
            args.output
        )


if __name__ == "__main__":
    main()