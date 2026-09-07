import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import subprocess
import json
import sys

from tkinterdnd2 import DND_FILES, TkinterDnD

from src.pose_processor import process_video, SUPPORTED_EXTENSIONS
from src.id_selection_window import IDSelectionWindow
from src.clean_data import clean_csv
from src.visualize_clean import create_visualization

import os

from src.paths import (
    BASE_DIR,
    VIDEOS_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    CONFIG_FILE
)

MODELS = {
    "YOLO26 Nano": "yolo26n-pose.pt",
    "YOLO26 Small": "yolo26s-pose.pt",
    "YOLO26 Medium": "yolo26m-pose.pt",
    "YOLO26 Large": "yolo26l-pose.pt",
    "YOLO26 XLarge": "yolo26x-pose.pt",
}

VIDEOS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def load_config():
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    
def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def show_video(video_path):
    video_path = Path(video_path).resolve()

    if sys.platform == "win32":
        os.startfile(video_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(video_path)])
    else:
        subprocess.run(["xdg-open", str(video_path)])

class PoseApp(TkinterDnD.Tk):

    def __init__(self):

        super().__init__()

        self.title("Analyse de mouvements")
        self.geometry("700x500")
        self.minsize(600, 500)

        self.video_name = None
        self.video_path = None

        self.baby_direction = None

        self.output_dir = None

        self.processing = False
        self.config = load_config()
        self.model_dir = Path(self.config.get("model_dir", BASE_DIR / "models"))

        self.create_widgets()

    def create_widgets(self):

        # ========================================================
        # Configuration
        # ========================================================

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_frame = tk.Frame(self)

        main_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=25,
            pady=15
        )

        main_frame.grid_columnconfigure(0, weight=1)

        # ========================================================
        # Titre
        # ========================================================

        title = tk.Label(
            main_frame,
            text="Analyse de mouvements du nourrisson",
            font=("Arial", 18, "bold")
        )

        title.grid(
            row=0,
            column=0,
            pady=(0, 3)
        )

        # ========================================================
        # Sous-titre
        # ========================================================

        subtitle = tk.Label(
            main_frame,
            text="Sélectionnez une vidéo pour lancer l'analyse.",
            font=("Arial", 10)
        )

        subtitle.grid(
            row=1,
            column=0,
            pady=(0, 8)
        )

        # ========================================================
        # Configuration modèle
        # ========================================================

        model_frame = tk.Frame(main_frame)

        model_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        model_frame.grid_columnconfigure(1, weight=1)

        tk.Label(
            model_frame,
            text="Dossier des modèles :"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8)
        )

        self.model_path_label = tk.Label(
            model_frame,
            text=str(self.model_dir),
            anchor="w"
        )

        self.model_path_label.grid(
            row=0,
            column=1,
            sticky="ew"
        )

        tk.Button(
            model_frame,
            text="Changer",
            command=self.select_model_folder
        ).grid(
            row=0,
            column=2,
            padx=(8, 0)
        )

        # ========================================================
        # Modèle YOLO
        # ========================================================

        model_select_frame = tk.Frame(main_frame)

        model_select_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        tk.Label(
            model_select_frame,
            text="Modèle YOLO :"
        ).pack(side="left")

        self.model_variable = tk.StringVar(
            value="YOLO26 Small"
        )

        self.model_combobox = ttk.Combobox(
            model_select_frame,
            textvariable=self.model_variable,
            values=list(MODELS.keys()),
            state="readonly",
            width=22
        )

        self.model_combobox.pack(
            side="left",
            padx=10
        )

        # ========================================================
        # Zone drag & drop
        # ========================================================

        self.drop_zone = tk.Label(
            main_frame,
            text="Glissez-déposez une vidéo ici\n\n"
                "ou cliquez pour sélectionner un fichier",
            relief="solid",
            borderwidth=2,
            font=("Arial", 11),
            cursor="hand2"
        )

        self.drop_zone.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=8,
            ipady=15
        )

        self.drop_zone.drop_target_register(DND_FILES)

        self.drop_zone.dnd_bind(
            "<<Drop>>",
            self.on_drop
        )

        self.drop_zone.bind(
            "<Button-1>",
            lambda event: self.select_video()
        )

        # ========================================================
        # Vidéo sélectionnée
        # ========================================================

        self.video_label = tk.Label(
            main_frame,
            text="Aucune vidéo sélectionnée",
            font=("Arial", 9),
            anchor="center"
        )

        self.video_label.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(3, 5)
        )

        # ========================================================
        # Bouton sélection vidéo
        # ========================================================

        self.select_button = tk.Button(
            main_frame,
            text="Sélectionner une vidéo",
            command=self.select_video,
            width=22,
            height=1
        )

        self.select_button.grid(
            row=6,
            column=0,
            pady=5
        )

        # ========================================================
        # Progression
        # ========================================================

        progress_frame = tk.Frame(main_frame)

        progress_frame.grid(
            row=7,
            column=0,
            sticky="ew",
            pady=(8, 3)
        )

        progress_frame.grid_columnconfigure(
            0,
            weight=1
        )

        self.progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate"
        )

        self.progress.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        self.progress_label = tk.Label(
            progress_frame,
            text="En attente",
            font=("Arial", 9)
        )

        self.progress_label.grid(
            row=1,
            column=0,
            pady=(2, 0)
        )

        # ========================================================
        # Bouton lancement
        # ========================================================

        self.start_button = tk.Button(
            main_frame,
            text="Lancer l'analyse",
            command=self.start_processing,
            width=22,
            height=1,
            state="disabled"
        )

        self.start_button.grid(
            row=8,
            column=0,
            pady=8
        )

        # ========================================================
        # Status
        # ========================================================

        self.status_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 10)
        )

        self.status_label.grid(
            row=9,
            column=0,
            pady=(0, 2)
        )

    def is_valid_video(self, path):

        return (
            Path(path).suffix.lower()
            in SUPPORTED_EXTENSIONS
        )

    def set_video(self, path):

        path = Path(path)

        if not self.is_valid_video(path):

            messagebox.showerror(
                "Format non supporté",
                (
                    "Veuillez sélectionner une vidéo "
                    "(.mp4, .avi, .mov ou .mkv)."
                )
            )

            return

        self.video_path = path
        self.video_name = path.stem
        if "g" in self.video_name.lower():
            self.baby_direction = "left"
        elif "d" in self.video_name.lower():
            self.baby_direction = "right"
        else:
            self.ask_baby_direction()

        self.output_dir = RESULTS_DIR / self.video_name

        direction_text = (
            "gauche" if self.baby_direction == "left" else "droite"
        )

        self.video_label.config(text=f"Vidéo sélectionnée : {self.video_name}\nDirection du bébé : {direction_text}")
        self.start_button.config(state="normal")
        self.status_label.config(text="Vidéo prête à être analysée.")

    def ask_baby_direction(self):
        window = tk.Toplevel(self)
        window.title("Direction du bébé")
        window.geometry("350x150")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()

        tk.Label(
            window,
            text="Dans quelle direction le bébé fait-il face ?",
            font=("Arial", 11)
        ).pack(pady=20)

        button_frame = tk.Frame(window)
        button_frame.pack()

        def select_direction(direction):
            self.baby_direction = direction
            window.destroy()

        tk.Button(
            button_frame,
            text="← Gauche",
            width=12,
            command=lambda: select_direction("left")
        ).pack(side="left", padx=10)

        tk.Button(
            button_frame,
            text="Droite →",
            width=12,
            command=lambda: select_direction("right")
        ).pack(side="left", padx=10)

        self.wait_window(window)

    def on_drop(self, event):

        paths = self.tk.splitlist(event.data)

        if paths:
            self.set_video(paths[0])

    def select_video(self):

        path = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            initialdir=VIDEOS_DIR,
            filetypes=[
                (
                    "Vidéos",
                    "*.mp4 *.avi *.mov *.mkv"
                ),
                (
                    "Tous les fichiers",
                    "*.*"
                )
            ]
        )

        if path:
            self.set_video(path)

    def select_model_folder(self):

        folder = filedialog.askdirectory(
            title="Sélectionner le dossier contenant les modèles YOLO"
        )

        if not folder:
            return

        self.model_dir = Path(folder)

        self.model_path_label.config(
            text=str(self.model_dir)
        )

        config = load_config()

        config["model_dir"] = str(self.model_dir)

        save_config(config)

    def start_processing(self):

        if self.video_path is None:
            return

        if self.processing:
            return

        baby_csv = self.output_dir / f"baby_{self.video_name}.csv"

        if self.output_dir.exists():

            answer = messagebox.askyesno(
                "Vidéo déjà analysée",
                (
                    "Cette vidéo a déjà été analysée.\n\n"
                    "Voulez-vous relancer l'analyse ?"
                )
            )

            if not answer:

                video_path = self.output_dir / f"visualization_{self.video_name}.mp4"

                if video_path.exists():
                    self.after(
                        0,
                        lambda: show_video(video_path)
                    )
                    return
                
                if baby_csv.exists():
                    self.after(
                        0,
                        lambda: self.process_baby_data(baby_csv)
                    )
                    return

                all_csv = self.output_dir / f"{self.video_name}.csv"
                if all_csv.exists():
                    self.after(
                        0,
                        lambda: self.processing_finished(all_csv)
                    )
                    return
                else:
                    messagebox.showwarning(
                        "CSV du bébé introuvable",
                        (
                            "La vidéo a déjà été analysée, "
                            "mais le fichier CSV du bébé est introuvable.\n\n"
                            "L'analyse YOLO va être relancée."
                        )
                    )

        if not self.baby_direction:
            self.ask_baby_direction()
        
        self.processing = True

        self.select_button.config(
            state="disabled"
        )

        self.start_button.config(
            state="disabled"
        )

        self.progress["value"] = 0

        self.progress_label.config(
            text="0 %"
        )

        self.status_label.config(
            text="Analyse YOLO en cours..."
        )

        thread = threading.Thread(
            target=self.run_processing,
            daemon=True
        )

        thread.start()

    def update_progress(self, value):

        self.after(
            0,
            lambda: self._update_progress(value)
        )

    def _update_progress(self, value):

        self.progress["value"] = value

        self.progress_label.config(
            text=f"{value:.0f} %"
        )

    def run_processing(self):
        """
        Exécute le traitement de la vidéo.
        """
        
        try:
            all_csv = process_video(
                self.video_path,
                direction=self.baby_direction,
                results_dir=RESULTS_DIR,
                model_path= self.model_dir / MODELS[self.model_variable.get()],
                progress_callback=self.update_progress
            )

            self.after(
                0,
                lambda: self.processing_finished(
                    all_csv
                )
            )
        except Exception as error:

            self.after(
                0,
                lambda error=error: self.processing_failed(
                    error
                )
            )

    def processing_finished(self, all_csv):
        """
        Affiche la fenêtre de sélection des IDs après la fin du traitement.
        """

        self.processing = False

        self.select_button.config(state="normal")
        self.progress["value"] = 100
        self.progress_label.config(text="100 %")
        self.status_label.config(text="Analyse YOLO terminée.")

        # Vidéo générée par YOLO
        generated_video = self.output_dir / f"{self.video_name}.mp4"

        # Vérifier que la vidéo existe
        if not generated_video.exists():

            self.start_button.config(state="normal")

            messagebox.showerror(
                "Vidéo introuvable",
                (
                    "L'analyse YOLO est terminée, "
                    "mais la vidéo annotée n'a pas été trouvée.\n\n"
                    f"Fichier attendu :\n{generated_video}"
                )
            )

            return

        # Ouvre la fenêtre de sélection des IDs
        self.status_label.config(
            text="Sélectionnez maintenant les IDs du bébé."
        )

        IDSelectionWindow(
            self,
            self.video_name,
            all_csv,
            on_validate=self.process_baby_data
        )

    def process_baby_data(self, baby_csv):
        """
        Nettoie les données et génère la vidéo de visualisation.
        """

        self.status_label.config(
            text="Traitement des données en cours..."
        )

        self.start_button.config(state="disabled")
        self.select_button.config(state="disabled")

        thread = threading.Thread(
            target=self._run_baby_processing,
            args=(baby_csv,),
            daemon=True
        )

        thread.start()

    def _run_baby_processing(self, baby_csv):

        try:

            self.after(
                0,
                lambda: self.status_label.config(
                    text="Nettoyage des données..."
                )
            )

            clean_csv(
                baby_csv,
                progress_callback=self.update_progress
            )

            self.after(
                0,
                lambda: self.status_label.config(
                    text="Génération de la vidéo..."
                )
            )

            video_path = create_visualization(
                input_folder=baby_csv.parent.name,
                progress_callback=self.update_progress
            )

            self.progress.stop()
            self.progress.config(
                mode="determinate",
                value=100
            )

            # Supprimer les fichiers intermédiaires
            files = [self.output_dir.glob(f"{self.video_name}.csv"), self.output_dir.glob(f"{self.video_name}.mp4"), self.output_dir.glob(f"baby_{self.video_name}.csv")]

            for file in files:
                for f in file:
                    f.unlink()

            self.after(
                0,
                lambda: show_video(
                    video_path
                )
            )

        except Exception as error:

            self.after(
                0,
                lambda error=error: self.processing_failed(
                    error
                )
            )

    def processing_failed(self, error):
        """
        Affiche un message d'erreur si le traitement échoue.
        """

        self.processing = False

        self.select_button.config(state="normal")
        self.start_button.config(state="normal")
        self.status_label.config(text="Une erreur est survenue.")

        messagebox.showerror(
            "Erreur",
            (
                "L'analyse n'a pas pu être terminée.\n\n"
                f"{error}"
            )
        )

if __name__ == "__main__":

    app = PoseApp()
    app.mainloop()