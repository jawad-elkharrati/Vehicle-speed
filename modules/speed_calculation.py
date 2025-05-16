"""
Module de calcul de la vitesse pour déterminer les vitesses des véhicules en fonction de la distance et du temps.
Ce module calcule les vitesses des véhicules détectés et suivis dans la vidéo.
"""

import time
import numpy as np
import cv2

class SpeedCalculator:
    def __init__(self, distance_calculator, fps=None):
        """
        Initialiser le calculateur de vitesse.
        
        Args:
            distance_calculator: Objet DistanceCalculator pour les mesures de distance
            fps: Frames par seconde de la vidéo (si None, sera calculé en temps réel)
        """
        self.distance_calculator = distance_calculator
        self.fps = fps
        self.vehicle_data = {}  # Dictionnaire pour stocker les données des véhicules pour le calcul de la vitesse
        self.last_frame_time = None
        self.speeds = {}  # Dictionnaire pour stocker les vitesses calculées pour chaque véhicule
    
    def set_fps(self, fps):
        """
        Définir la valeur des frames par seconde.
        
        Args:
            fps: Frames par seconde de la vidéo
        """
        self.fps = fps
    
    def update(self, vehicles, frame_number=None, current_time=None):
        """
        Mettre à jour le calculateur de vitesse avec les nouvelles positions des véhicules.
        
        Args:
            vehicles: Dictionnaire des véhicules suivis avec leurs IDs et boîtes englobantes
            frame_number: Numéro de la frame actuelle (utilisé si fps est fourni)
            current_time: Temps actuel en secondes (utilisé si fps n'est pas fourni)
            
        Retourne:
            Dictionnaire des IDs des véhicules et leurs vitesses calculées
        """
        # Calculer le delta de temps
        if self.fps is not None and frame_number is not None:
            # Utiliser le numéro de la frame et le fps pour le calcul du temps
            current_frame_time = frame_number / self.fps
        else:
            # Utiliser l'heure système si fps n'est pas fourni
            current_frame_time = current_time if current_time is not None else time.time()
        
        # Initialiser last_frame_time si c'est la première mise à jour
        if self.last_frame_time is None:
            self.last_frame_time = current_frame_time
            
        time_delta = current_frame_time - self.last_frame_time
        self.last_frame_time = current_frame_time
        
        # Mettre à jour les données des véhicules et calculer les vitesses
        for vehicle_id, vehicle_data in vehicles.items():
            bbox = vehicle_data['bbox']
            
            # Obtenir le point central du véhicule
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            current_position = (center_x, center_y)
            
            # Si c'est un nouveau véhicule, initialiser ses données
            if vehicle_id not in self.vehicle_data:
                self.vehicle_data[vehicle_id] = {
                    'positions': [current_position],
                    'times': [current_frame_time],
                    'speeds': []
                }
            else:
                # Obtenir la position précédente
                prev_position = self.vehicle_data[vehicle_id]['positions'][-1]
                
                # Calculer la distance parcourue en mètres
                distance_meters = self.distance_calculator.calculate_distance(prev_position, current_position)
                
                # Calculer le temps écoulé en secondes
                if len(self.vehicle_data[vehicle_id]['times']) > 0:
                    time_elapsed = current_frame_time - self.vehicle_data[vehicle_id]['times'][-1]
                else:
                    time_elapsed = time_delta
                
                # Calculer la vitesse en mètres par seconde
                if time_elapsed > 0:
                    speed_mps = distance_meters / time_elapsed
                    
                    # Convertir en km/h
                    speed_kmh = speed_mps * 3.6
                    
                    # Ajouter à la liste des vitesses
                    self.vehicle_data[vehicle_id]['speeds'].append(speed_kmh)
                    
                    # Calculer la vitesse moyenne sur les dernières mesures
                    num_measurements = min(5, len(self.vehicle_data[vehicle_id]['speeds']))
                    avg_speed = np.mean(self.vehicle_data[vehicle_id]['speeds'][-num_measurements:])
                    
                    # Stocker la vitesse moyenne
                    self.speeds[vehicle_id] = avg_speed
                
                # Mettre à jour la position et le temps
                self.vehicle_data[vehicle_id]['positions'].append(current_position)
                self.vehicle_data[vehicle_id]['times'].append(current_frame_time)
        
        # Supprimer les véhicules qui ne sont plus suivis
        for vehicle_id in list(self.vehicle_data.keys()):
            if vehicle_id not in vehicles:
                # Garder les données de vitesse mais supprimer l'historique des positions
                if vehicle_id in self.speeds:
                    final_speed = self.speeds[vehicle_id]
                    self.speeds[vehicle_id] = final_speed
                self.vehicle_data.pop(vehicle_id, None)
        
        return self.speeds
    
    def draw_speeds(self, frame, vehicles):
        """
        Dessiner les informations de vitesse sur l'image.
        
        Args:
            frame: Image sur laquelle dessiner
            vehicles: Dictionnaire des véhicules suivis avec leurs IDs et boîtes englobantes
            
        Retourne:
            Image avec les informations de vitesse dessinées
        """
        result_frame = frame.copy()
        
        for vehicle_id, vehicle_data in vehicles.items():
            if vehicle_id in self.speeds:
                # Obtenir la boîte englobante du véhicule
                x, y, w, h = vehicle_data['bbox']
                
                # Dessiner le texte de la vitesse au-dessus de la boîte englobante
                speed_text = f"ID: {vehicle_id}, {self.speeds[vehicle_id]:.1f} km/h"
                cv2.putText(result_frame, speed_text, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return result_frame
    
    def get_average_speed(self):
        """
        Obtenir la vitesse moyenne de tous les véhicules suivis.
        
        Retourne:
            Vitesse moyenne en km/h
        """
        if not self.speeds:
            return 0
        
        return np.mean(list(self.speeds.values()))
    
    def get_speed_data(self):
        """
        Obtenir toutes les données de vitesse.
        
        Retourne:
            Dictionnaire des IDs des véhicules et leurs vitesses
        """
        return self.speeds
