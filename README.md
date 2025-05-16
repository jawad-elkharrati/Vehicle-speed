# Système de Détection de Vitesse des Véhicules

Un système basé sur Python pour détecter, suivre et mesurer la vitesse des véhicules dans des séquences vidéo capturées par une caméra fixe.

## Réalisé par :
- **ELKHARRATI JAWAD**
- **NUKUNU SHALOM JUNIOR**
- **CHERIF HANAE**
- **GUINI YOUSSEF**

## Vue d'Ensemble

Ce système traite les séquences vidéo provenant d'une caméra fixe surplombant une route pour :
- Détecter les véhicules dans chaque image
- Suivre les véhicules à travers plusieurs images
- Calculer la vitesse de chaque véhicule
- Compter les véhicules uniques
- Stocker les données de détection et de vitesse pour une analyse ultérieure

Le système utilise des techniques de vision par ordinateur avec OpenCV pour une mise en œuvre légère pouvant fonctionner sur des systèmes avec des ressources limitées.

## Structure du Projet
vehicle_speed_detection/
├── modules/
│   ├── object_detection.py       # Détection des véhicules via soustraction de fond
│   ├── vehicle_tracking.py       # Suivi des véhicules à travers les images
│   ├── distance_calculation.py   # Conversion entre distances en pixels et réelles
│   ├── speed_calculation.py      # Calcul de la vitesse basé sur la distance et le temps
│   ├── vehicle_counting.py       # Comptage des véhicules uniques
│   └── data_storage.py           # Stockage des données de détection et de vitesse
├── data/                         # Répertoire pour les fichiers de données de sortie
├── main.py                      # Application principale
└── requirements.txt             # Dépendances nécessaires

## Installez les dépendances nécessaires :
    pip install -r requirements.txt
## Utilisation
 -- Interface en Ligne de Commande --
  Le système peut être exécuté depuis la ligne de commande avec diverses options :
- ** python main.py --input <input-video> --output <output-video> --mode video --display **
## Options :
--input : Chemin vers le fichier vidéo d'entrée

--output : Chemin vers le fichier vidéo de sortie (optionnel)

--mode : Mode de traitement (vidéo, image, ou caméra)

--camera : ID de la caméra pour le mode caméra (par défaut : 0)

--display : Afficher les résultats dans une fenêtre

--lane-width : Largeur de la voie de référence en mètres (par défaut : 3.5)

--detection-line : Position de la ligne de détection (0-1) (par défaut : 0.6)

--min-area : Aire minimale d'un véhicule en pixels (par défaut : 500)

--output-dir : Répertoire de sortie pour les fichiers de données (par défaut : data)

### Exemples
Traiter un fichier vidéo :
python main.py --input traffic_video.mp4 --output processed_video.avi --mode video --display

Traiter un flux de caméra :
python main.py --mode camera --camera 0 --display

Traiter une image :
python main.py --input traffic_image.jpg --output processed_image.jpg --mode image

###     Modules
##  Détection des Objets
Le module de détection des objets utilise la soustraction de fond et la détection de contours pour identifier les véhicules dans chaque image. Il est implémenté dans modules/object_detection.py.

#   Principales caractéristiques :

Soustraction de fond utilisant l'algorithme MOG2

Opérations morphologiques pour l'élimination du bruit

Détection de contours et filtrage basé sur la surface

Ligne de détection pour le comptage des véhicules

##  Suivi des Véhicules
Le module de suivi des véhicules suit les véhicules détectés à travers plusieurs images. Il est implémenté dans modules/vehicle_tracking.py.

#   Principales caractéristiques :

Calcul du taux de recouvrement des boîtes englobantes

Attribution d'ID de véhicules et suivi

Gestion de la disparition et de la réapparition des véhicules

Détection de la traversée de la ligne de détection

##  Calcul de la Distance
Le module de calcul de la distance gère la conversion entre les distances en pixels et les mesures réelles. Il est implémenté dans modules/distance_calculation.py.

#   Principales caractéristiques :

Conversion des pixels en mètres

Calibration à l'aide d'un objet de référence

Calcul de la distance entre deux points

Calibration basée sur la largeur de la voie

##  Calcul de la Vitesse
Le module de calcul de la vitesse calcule les vitesses des véhicules en fonction des mesures de distance et de temps. Il est implémenté dans modules/speed_calculation.py.

#   Principales caractéristiques :

Calcul de la vitesse en mètres par seconde et en km/h

Calcul de la vitesse moyenne sur plusieurs images

Visualisation de la vitesse sur les images vidéo

Stockage des données de vitesse pour chaque véhicule

##  Comptage des Véhicules
Le module de comptage des véhicules compte les véhicules uniques qui traversent une ligne de détection. Il est implémenté dans modules/vehicle_counting.py.

#   Principales caractéristiques :

Détection de la traversée de la ligne de détection

Comptage unique des véhicules

Calcul du taux de comptage

Visualisation du comptage sur les images vidéo

##  Stockage des Données
Le module de stockage des données sauvegarde les données de détection, de suivi et de vitesse des véhicules pour une analyse ultérieure. Il est implémenté dans modules/data_storage.py.

#   Principales caractéristiques :

Exportation des données au format CSV et JSON

Génération de statistiques récapitulatives

Calcul de la distribution des vitesses

Organisation des données par session

##  Configuration
Le système peut être configuré via les arguments de la ligne de commande ou en modifiant le dictionnaire de configuration dans la classe VehicleSpeedDetectionSystem.

#   Options de configuration principales :

min_vehicle_area : Aire minimale d'un véhicule pour être considéré

detection_line_position : Position de la ligne de détection (0-1)

reference_width_meters : Largeur de l'objet de référence en mètres

output_dir : Répertoire pour sauvegarder les fichiers de données

save_interval : Intervalle de sauvegarde des données en secondes

display_results : Si les résultats doivent être affichés dans une fenêtre

*** Données de Sortie ***
Le système génère plusieurs fichiers de données :

Fichier CSV des données des véhicules : Contient des informations détaillées sur chaque détection de véhicule

Fichier CSV récapitulatif : Contient les statistiques récapitulatives pour la session

Fichier CSV de la distribution des vitesses : Contient la distribution des vitesses des véhicules

Ces fichiers sont sauvegardés dans le répertoire de sortie spécifié.

*** Limitations et Améliorations Futures ***
### Limitations actuelles :
Le système suppose une caméra fixe

Le calcul de la distance repose sur une méthode de calibration simple

La détection des objets utilise la soustraction de fond, ce qui peut ne pas fonctionner dans toutes les conditions


### Améliorations potentielles :
    Implémenter une détection d'objets basée sur l'apprentissage profond (YOLO, SSD, etc.)

    Améliorer le suivi avec des algorithmes plus sophistiqués (DeepSORT, etc.)

    Ajouter une calibration de caméra pour de meilleures mesures de distance

    Implémenter un support multi-caméras

    Ajouter une interface web pour la surveillance en temps réel

-*** OpenCV pour la fonctionnalité de vision par ordinateur ***

-*** NumPy pour les opérations numériques ***

-*** Pandas pour la manipulation des données ***
