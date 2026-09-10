# Analyse de mouvements du nourrisson

Application permettant d'analyser une vidéo avec YOLO26 Pose et de générer les positions des keypoints dans un fichier CSV.

# Arborescence

```
.
├── README.md             # Explication du projet
├── app.py                # Interface principale
├── aruco_markers         # Ressources des marqueurs ArUco
├── models                # Modèles YOLO
├── movements.md          # Documentation des mouvements
├── requirements.txt      # Dépendances Python
├── results               # Résultats CSV, vidéo et graphes
├── scripts               # Scripts utilitaires
│   ├── generate_markers.py          # Génération des marqueurs
│   ├── generate_smoothed_graph.py   # Génération des graphes lissés
│   └── make_sheet.py                # Création de la planche de marqueurs
├── src                   # Code source principal
│   ├── clean_data.py               # Nettoyage des keypoints
│   ├── id_selection_window.py      # Sélection des IDs
│   ├── movement_detection.py       # Détection des mouvements (branche movements)
│   ├── paths.py                    # Gestion des chemins
│   ├── pose_processor.py           # Post-traitement des poses
│   └── visualize_clean.py          # Visualisation des données
└── videos                # Vidéos à analyser
```
---

# Installation

## 1. Installation

Python 3.10 ou supérieur est recommandé.

Cloner le repository :
```sh
git clone https://github.com/Perinat-Cognition/BabyMove.git
cd BabyMove
```

Créer un environnement virtuel :

```sh
python -m venv .venv
```

Activer l'environnement :

Windows :

```sh
.venv\Scripts\activate
```

Installer les dépendances :

```sh
pip install -r requirements.txt
```

## 3. Lancer l'application

Depuis le dossier principal du projet :

```sh
python app.py
```

L'interface permet ensuite de sélectionner une vidéo dans `videos/` ou de la glisser-déposer dans l'application.

Une fois l'analyse terminée, les résultats sont enregistrés dans `results/`.

## 4. Modèle YOLO

L'application utilise actuellement YOLO26 Pose pour détecter les 17 keypoints humains.

Les modèles YOLO sont placés dans `models/` par défaut. Le chemin du modèle peut être modifié dans l'application et est sauvegardé dans `config.json`. 
Le modèle sera automatiquement téléchargé si le fichier n'existe pas.

## 5. Marqueurs ArUco

Pour générer la planche de marqueurs ArUco avec une taille par défaut de 20 mm :

```sh
python -m scripts.make_sheet
```

Pour choisir la taille des marqueurs en millimètres :

```sh
python -m scripts.make_sheet --size 10
```