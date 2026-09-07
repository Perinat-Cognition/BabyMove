import sys
import csv
import cv2
import os
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageOps

from src.paths import (
    RESULTS_DIR
)

def get_ids_from_csv(csv_path):
    """Return all person IDs found in the CSV."""

    ids = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                person_id = int(float(row["person_id"]))

                if person_id >= 0:
                    ids.add(person_id)

            except (ValueError, TypeError):
                pass

    return sorted(ids)


def create_baby_csv(all_csv, baby_ids, output_csv):
    """
    Create a CSV containing only the selected baby IDs.
    """

    baby_ids = set(baby_ids)

    with open(
        all_csv,
        "r",
        encoding="utf-8"
    ) as infile:

        reader = csv.DictReader(infile)

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as outfile:

            writer = csv.DictWriter(
                outfile,
                fieldnames=reader.fieldnames
            )

            writer.writeheader()

            for row in reader:

                try:
                    person_id = int(float(row["person_id"]))
                except (ValueError, TypeError):
                    continue

                if person_id in baby_ids:
                    writer.writerow(row)

class IDSelectionWindow(tk.Toplevel):

    def __init__(
        self,
        parent,
        output_folder,
        all_csv,
        on_validate=None
    ):
        super().__init__(parent)

        self.parent = parent
        self.all_csv = Path(all_csv)
        self.output_dir = RESULTS_DIR / output_folder
        self.video_path = self.output_dir / f"{output_folder}.mp4"

        self.title("Sélection du bébé")
        self.geometry("1100x800")
        self.minsize(800, 800)

        self.on_validate = on_validate

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            messagebox.showerror(
                "Erreur",
                "Impossible d'ouvrir la vidéo."
            )
            self.destroy()
            return

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if self.fps <= 0:
            self.fps = 25

        self.current_frame = 0
        self.playing = False

        self.photo = None

        self.ids = get_ids_from_csv(
            self.all_csv
        )

        self.id_variables = {}

        self.create_widgets()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.close_window
        )

        self.show_frame(0)

    def resize_video(self, event):

        if self.current_frame is not None:
            self.show_frame(
                self.current_frame
            )

    # --------------------------------------------------------
    # INTERFACE
    # --------------------------------------------------------

    def create_widgets(self):

        title = tk.Label(
            self,
            text="Sélection du bébé",
            font=("Arial", 18, "bold")
        )

        title.pack(pady=(15, 5))

        instruction = tk.Label(
            self,
            text=(
                "Sélectionnez tous les IDs correspondant au bébé. "
                "Si le bébé reçoit un nouvel ID après une sortie du champ, "
                "sélectionnez également cet ID."
            ),
            font=("Arial", 10),
            wraplength=900
        )

        instruction.pack(pady=(0, 10))

        # ----------------------------------------------------
        # Zone vidéo
        # ----------------------------------------------------

        self.video_label = tk.Label(
            self,
            bg="black",
            width=900,
            height=500
        )

        self.video_label.pack(
            padx=15,
            pady=10
        )

        self.video_label.bind(
            "<Configure>",
            self.resize_video
        )

        # ----------------------------------------------------
        # Contrôles vidéo
        # ----------------------------------------------------

        controls = tk.Frame(self)

        controls.pack(
            fill="x",
            padx=20
        )

        self.previous_button = tk.Button(
            controls,
            text="◀ Frame",
            command=self.previous_frame,
            width=10
        )

        self.previous_button.pack(
            side="left",
            padx=5
        )

        self.play_button = tk.Button(
            controls,
            text="▶ Lecture",
            command=self.toggle_play,
            width=12
        )

        self.play_button.pack(
            side="left",
            padx=5
        )

        self.next_button = tk.Button(
            controls,
            text="Frame ▶",
            command=self.next_frame,
            width=10
        )

        self.next_button.pack(
            side="left",
            padx=5
        )

        self.frame_label = tk.Label(
            controls,
            text=""
        )

        self.frame_label.pack(
            side="right",
            padx=10
        )

        # ----------------------------------------------------
        # Barre de progression vidéo
        # ----------------------------------------------------

        self.frame_scale = tk.Scale(
            self,
            from_=0,
            to=max(0, self.total_frames - 1),
            orient="horizontal",
            showvalue=False,
            command=self.seek_frame
        )

        self.frame_scale.pack(
            fill="x",
            padx=25
        )

        # ----------------------------------------------------
        # Sélection des IDs
        # ----------------------------------------------------

        id_frame = tk.LabelFrame(
            self,
            text="IDs détectés",
            padx=10,
            pady=10
        )

        id_frame.pack(
            fill="x",
            padx=20,
            pady=10
        )

        if not self.ids:

            tk.Label(
                id_frame,
                text="Aucun ID détecté."
            ).pack()

        else:

            for person_id in self.ids:

                variable = tk.BooleanVar(
                    value=False
                )

                self.id_variables[
                    person_id
                ] = variable

                checkbox = tk.Checkbutton(
                    id_frame,
                    text=f"ID {person_id}",
                    variable=variable,
                    font=("Arial", 11)
                )

                checkbox.pack(
                    side="left",
                    padx=10
                )

        bottom_frame = tk.Frame(self)

        bottom_frame.pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        self.validate_button = tk.Button(
            bottom_frame,
            text="Valider les IDs du bébé",
            command=self.validate,
            width=25,
            height=2
        )

        self.validate_button.pack(
            side="right"
        )

    def show_frame(self, frame_number):

        if self.total_frames <= 0:
            return

        frame_number = max(
            0,
            min(
                frame_number,
                self.total_frames - 1
            )
        )

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = self.cap.read()

        if not success:
            return

        self.current_frame = frame_number

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        image = ImageOps.contain(
            image,
            (900, 500)
        )

        self.photo = ImageTk.PhotoImage(image)

        self.video_label.config(
            image=self.photo
        )

        self.frame_label.config(
            text=(
                f"Frame {frame_number + 1} / "
                f"{self.total_frames}"
            )
        )

        self.frame_scale.set(frame_number)

    def toggle_play(self):

        if self.playing:

            self.playing = False

            self.play_button.config(
                text="▶ Lecture"
            )

        else:

            self.playing = True

            self.play_button.config(
                text="⏸ Pause"
            )

            self.play_video()

    def play_video(self):

        if not self.playing:
            return

        if self.current_frame >= self.total_frames - 1:

            self.playing = False

            self.play_button.config(
                text="▶ Lecture"
            )

            return

        self.show_frame(
            self.current_frame + 1
        )

        delay = max(
            10,
            int(1000 / self.fps)
        )

        self.after(
            delay,
            self.play_video
        )

    def previous_frame(self):

        self.playing = False

        self.play_button.config(
            text="▶ Lecture"
        )

        self.show_frame(
            self.current_frame - 1
        )

    def next_frame(self):

        self.playing = False

        self.play_button.config(
            text="▶ Lecture"
        )

        self.show_frame(
            self.current_frame + 1
        )

    def seek_frame(self, value):

        try:
            frame_number = int(float(value))
        except ValueError:
            return

        # Ne pas faire de seek si on est déjà
        # sur cette frame.
        if frame_number != self.current_frame:
            self.show_frame(
                frame_number
            )

    def get_selected_ids(self):

        return [
            person_id
            for person_id, variable
            in self.id_variables.items()
            if variable.get()
        ]

    def validate(self):

        selected_ids = self.get_selected_ids()

        if not selected_ids:

            answer = messagebox.askyesno(
                "Aucun ID sélectionné",
                (
                    "Aucun ID n'a été sélectionné.\n\n"
                    "Voulez-vous vraiment créer un CSV vide ?"
                ),
                parent=self
            )

            if not answer:
                return

        output_csv = (
            self.output_dir /
            f"baby_{self.output_dir.name}.csv"
        )

        try:

            create_baby_csv(
                self.all_csv,
                selected_ids,
                output_csv
            )

        except Exception as error:

            messagebox.showerror(
                "Erreur",
                f"Impossible de créer le CSV :\n\n{error}",
                parent=self
            )

            return

        self.playing = False
        self.cap.release()

        self.destroy()

        # Lance l'étape suivante uniquement après validation
        if self.on_validate is not None:
            self.on_validate(output_csv)

    def close_window(self):

        self.playing = False

        if self.cap:
            self.cap.release()

        self.destroy()