"""
Application principale pour le système de détection de vitesse des véhicules.
Ce module intègre tous les composants dans un système complet.
"""

import cv2
import numpy as np
import time
import os
import argparse
from datetime import datetime

# Importer les modules
from modules.object_detection import VehicleDetector
from modules.vehicle_tracking import VehicleTracker
from modules.distance_calculation import DistanceCalculator
from modules.speed_calculation import SpeedCalculator
from modules.vehicle_counting import VehicleCounter
from modules.data_storage import DataStorage

class VehicleSpeedDetectionSystem:
    def __init__(self, config=None):
        """
        Initialiser le système de détection de vitesse des véhicules.
        
        Args:
            config: Dictionnaire de configuration (si None, utilise la configuration par défaut)
        """
        # Configuration par défaut
        self.config = {
            'min_vehicle_area': 500,
            'detection_line_position': 0.6,
            'reference_width_meters': 3.5,  # Largeur standard de la voie
            'output_dir': 'data',
            'save_interval': 10,  # Sauvegarder les données toutes les 10 secondes
            'display_results': True
        }
        
        # Mettre à jour avec la configuration fournie
        if config is not None:
            self.config.update(config)
        
        # Créer le répertoire de sortie
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialiser les modules
        self.detector = VehicleDetector(
            min_area=self.config['min_vehicle_area'],
            detection_line_position=self.config['detection_line_position']
        )
        
        self.tracker = VehicleTracker(
            max_disappeared=10,
            min_overlap_ratio=0.3
        )
        
        self.distance_calculator = DistanceCalculator()
        
        self.speed_calculator = SpeedCalculator(
            distance_calculator=self.distance_calculator
        )
        
        self.counter = VehicleCounter(
            detection_line_position=self.config['detection_line_position']
        )
        
        self.data_storage = DataStorage(
            output_dir=self.config['output_dir']
        )
        
        # Variables d'état
        self.frame_count = 0
        self.fps = 0
        self.last_save_time = time.time()
        self.processing_times = []
    
    def process_frame(self, frame, frame_time=None):
        """
        Traiter une seule image de la vidéo.
        
        Args:
            frame: Image d'entrée de la vidéo
            frame_time: Horodatage de l'image (si None, utilise l'heure actuelle)
            
        Retourne:
            Image traitée avec des visualisations
        """
        if frame_time is None:
            frame_time = time.time()
        
        start_time = time.time()
        
        # Étape 1 : Détecter les véhicules
        bounding_boxes, detection_frame, detection_line_y = self.detector.detect_vehicles(frame)
        
        # Étape 2 : Suivre les véhicules
        tracked_vehicles = self.tracker.update(bounding_boxes, detection_line_y)
        
        # Étape 3 : Calculer les vitesses (si le calculateur de distance est calibré)
        if self.distance_calculator.pixels_per_meter is None:
            # Calibrer le calculateur de distance si ce n'est pas déjà fait
            self.distance_calculator.calibrate_from_lane_width(
                frame, 
                lane_width_meters=self.config['reference_width_meters']
            )
        
        speeds = self.speed_calculator.update(
            tracked_vehicles, 
            frame_number=self.frame_count,
            current_time=frame_time
        )
        
        # Étape 4 : Mettre à jour le compteur de véhicules
        vehicle_count = self.counter.update(
            tracked_vehicles, 
            frame.shape,
            current_time=frame_time
        )
        
        # Étape 5 : Stocker les données
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            bbox = vehicle_data['bbox']
            speed = speeds.get(vehicle_id, None)
            crossed_line = vehicle_data.get('crossed_line', False)
            
            self.data_storage.add_vehicle_record(
                vehicle_id=vehicle_id,
                timestamp=frame_time,
                bbox=bbox,
                speed=speed,
                crossed_line=crossed_line
            )
        
        # Sauvegarder les données périodiquement
        if time.time() - self.last_save_time > self.config['save_interval']:
            self.save_data()
            self.last_save_time = time.time()
        
        # Calculer le temps de traitement et le FPS
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Calculer le FPS moyen sur les 10 derniers frames
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        self.fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        # Créer l'image résultat avec les visualisations
        if self.config['display_results']:
            # Dessiner les informations de suivi
            result_frame = detection_frame.copy()
            
            # Dessiner les vitesses
            result_frame = self.speed_calculator.draw_speeds(result_frame, tracked_vehicles)
            
            # Dessiner le compteur
            result_frame = self.counter.draw_counter(result_frame)
            
            # Dessiner la ligne de référence pour la distance
            result_frame = self.distance_calculator.draw_reference_line(result_frame)
            
            # Dessiner le FPS
            cv2.putText(result_frame, f"FPS: {self.fps:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dessiner l'horodatage
            timestamp_str = datetime.fromtimestamp(frame_time).strftime("%M:%S")
            cv2.putText(result_frame, timestamp_str, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            result_frame = frame
        
        self.frame_count += 1
        
        return result_frame
    
    def save_data(self):
        """
        Sauvegarder toutes les données collectées dans des fichiers.
        
        Retourne:
            Dictionnaire des chemins des fichiers sauvegardés
        """
        saved_files = {}
        
        # Sauvegarder les données des véhicules dans un fichier CSV
        csv_path = self.data_storage.save_to_csv()
        saved_files['csv'] = csv_path
        
        # Sauvegarder le résumé
        avg_speed = self.speed_calculator.get_average_speed()
        vehicle_count = self.counter.get_count()
        
        summary_path = self.data_storage.save_summary(
            vehicle_count=vehicle_count,
            avg_speed=avg_speed
        )
        saved_files['summary'] = summary_path
        
        # Sauvegarder la distribution des vitesses
        distribution_path = self.data_storage.save_speed_distribution()
        saved_files['distribution'] = distribution_path
        
        return saved_files
    
    def process_video(self, video_path, output_path=None):
        """
        Traiter un fichier vidéo.
        
        Args:
            video_path: Chemin vers le fichier vidéo d'entrée
            output_path: Chemin pour sauvegarder la vidéo de sortie (si None, ne sauvegarde pas)
            
        Retourne:
            Dictionnaire des chemins des fichiers de données sauvegardés
        """
        # Ouvrir le fichier vidéo
        cap = cv2.VideoCapture(video_path)
        
        # Vérifier si la vidéo est bien ouverte
        if not cap.isOpened():
            print(f"Erreur : Impossible d'ouvrir le fichier vidéo {video_path}")
            return {}
        
        # Obtenir les propriétés de la vidéo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Définir le FPS pour le calculateur de vitesse
        self.speed_calculator.set_fps(fps)
        
        # Créer un écrivain vidéo si un chemin de sortie est fourni
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Traiter les images de la vidéo
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Obtenir l'horodatage de l'image
            frame_time = self.frame_count / fps if fps > 0 else time.time()
            
            # Traiter l'image
            result_frame = self.process_frame(frame, frame_time)
            
            # Écrire l'image dans la vidéo de sortie
            if output_path is not None:
                out.write(result_frame)
            
            # Afficher l'image
            if self.config['display_results']:
                cv2.imshow('Détection de Vitesse des Véhicules', result_frame)
                
                # Sortir si 'q' est pressé
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Libérer les ressources
        cap.release()
        
        if output_path is not None:
            out.release()
        
        cv2.destroyAllWindows()
        
        # Sauvegarder les données finales
        saved_files = self.save_data()
        
        return saved_files
    
    def process_image(self, image_path, output_path=None):
        """
        Traiter une seule image.
        
        Args:
            image_path: Chemin vers le fichier image d'entrée
            output_path: Chemin pour sauvegarder l'image de sortie (si None, ne sauvegarde pas)
            
        Retourne:
            Image traitée
        """
        # Lire l'image
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Erreur : Impossible de lire le fichier image {image_path}")
            return None
        
        # Traiter l'image
        result_frame = self.process_frame(frame)
        
        # Sauvegarder l'image de sortie
        if output_path is not None:
            cv2.imwrite(output_path, result_frame)
        
        return result_frame
    
    def process_camera(self, camera_id=0, output_path=None, duration=None):
        """
        Traiter une vidéo depuis une caméra.
        
        Args:
            camera_id: ID de la caméra (par défaut : 0 pour la caméra par défaut)
            output_path: Chemin pour sauvegarder la vidéo de sortie (si None, ne sauvegarde pas)
            duration: Durée d'enregistrement en secondes (si None, s'arrête quand 'q' est pressé)
            
        Retourne:
            Dictionnaire des chemins des fichiers de données sauvegardés
        """
        # Ouvrir la caméra
        cap = cv2.VideoCapture(camera_id)
        
        # Vérifier si la caméra est bien ouverte
        if not cap.isOpened():
            print(f"Erreur : Impossible d'ouvrir la caméra {camera_id}")
            return {}
        
        # Obtenir les propriétés de la caméra
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Définir le FPS pour le calculateur de vitesse
        self.speed_calculator.set_fps(fps)
        
        # Créer un écrivain vidéo si un chemin de sortie est fourni
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Définir l'heure de début pour le suivi de la durée
        start_time = time.time()
        
        # Traiter les images de la caméra
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Traiter l'image
            result_frame = self.process_frame(frame)
            
            # Écrire l'image dans la vidéo de sortie
            if output_path is not None:
                out.write(result_frame)
            
            # Afficher l'image
            if self.config['display_results']:
                cv2.imshow('Détection de Vitesse des Véhicules', result_frame)
                
                # Sortir si 'q' est pressé
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Vérifier si la durée est écoulée
            if duration is not None and time.time() - start_time > duration:
                break
        
        # Libérer les ressources
        cap.release()
        
        if output_path is not None:
            out.release()
        
        cv2.destroyAllWindows()
        
        # Sauvegarder les données finales
        saved_files = self.save_data()
        
        return saved_files


def main():
    """
    Fonction principale pour exécuter le système de détection de vitesse des véhicules.
    """
    # Analyser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description='Système de Détection de Vitesse des Véhicules')
    parser.add_argument('--input', type=str, help='Chemin vers le fichier vidéo d\'entrée')
    parser.add_argument('--output', type=str, help='Chemin vers le fichier vidéo de sortie')
    parser.add_argument('--camera', type=int, default=0, help='ID de la caméra (par défaut : 0)')
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'image', 'camera'],
                        help='Mode de traitement (par défaut : vidéo)')
    parser.add_argument('--display', action='store_true', help='Afficher les résultats')
    parser.add_argument('--lane-width', type=float, default=1.5, 
                        help='Largeur de la voie de référence en mètres (par défaut : 3.5)')
    parser.add_argument('--detection-line', type=float, default=0.6, 
                        help='Position de la ligne de détection (0-1) (par défaut : 0.6)')
    parser.add_argument('--min-area', type=int, default=500, 
                        help='Surface minimale du véhicule en pixels (par défaut : 500)')
    parser.add_argument('--output-dir', type=str, default='data', 
                        help='Répertoire de sortie pour les fichiers de données (par défaut : data)')
    
    args = parser.parse_args()
    
    # Créer la configuration
    config = {
        'min_vehicle_area': args.min_area,
        'detection_line_position': args.detection_line,
        'reference_width_meters': args.lane_width,
        'output_dir': args.output_dir,
        'display_results': args.display
    }
    
    # Créer le système de détection de vitesse des véhicules
    system = VehicleSpeedDetectionSystem(config)
    
    # Traitement en fonction du mode
    if args.mode == 'video':
        if args.input is None:
            print("Erreur : Fichier vidéo d'entrée non spécifié")
            return
        
        print(f"Traitement de la vidéo : {args.input}")
        saved_files = system.process_video(args.input, args.output)
        
        print("Traitement terminé")
        print(f"Données sauvegardées dans : {saved_files}")
    
    elif args.mode == 'image':
        if args.input is None:
            print("Erreur : Fichier image d'entrée non spécifié")
            return
        
        print(f"Traitement de l'image : {args.input}")
        result_image = system.process_image(args.input, args.output)
        
        if result_image is not None:
            print("Traitement terminé")
            if args.output:
                print(f"Image de sortie sauvegardée dans : {args.output}")
    
    elif args.mode == 'camera':
        print(f"Traitement du flux de la caméra {args.camera}")
        saved_files = system.process_camera(args.camera, args.output)
        
        print("Traitement terminé")
        print(f"Données sauvegardées dans : {saved_files}")


if __name__ == "__main__":
    main()
