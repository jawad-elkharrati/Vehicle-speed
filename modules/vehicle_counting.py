"""
Module de comptage des véhicules pour suivre le nombre de véhicules uniques détectés.
Ce module compte les véhicules qui traversent une ligne de détection spécifiée.
"""

import cv2
import numpy as np
import time

class VehicleCounter:
    def __init__(self, detection_line_position=0.6):
        """
        Initialiser le compteur de véhicules.
        
        Args:
            detection_line_position (float): Position de la ligne de détection (0-1)
        """
        self.detection_line_position = detection_line_position
        self.vehicle_count = 0
        self.counted_vehicles = set()  # Ensemble des IDs des véhicules qui ont été comptés
        self.count_history = []  # Liste des tuples (horodatage, comptage) pour une analyse basée sur le temps
        self.last_update_time = time.time()
    
    def update(self, vehicles, frame_shape, current_time=None):
        """
        Mettre à jour le compteur de véhicules avec les nouvelles positions des véhicules.
        
        Args:
            vehicles: Dictionnaire des véhicules suivis avec leurs IDs et boîtes englobantes
            frame_shape: Forme de l'image vidéo (hauteur, largeur)
            current_time: Temps actuel en secondes (si None, utilise l'heure système)
            
        Retourne:
            Nombre actuel de véhicules
        """
        if current_time is None:
            current_time = time.time()
        
        # Calculer la coordonnée y de la ligne de détection
        line_y = int(frame_shape[0] * self.detection_line_position)
        
        # Vérifier si des véhicules ont traversé la ligne
        for vehicle_id, vehicle_data in vehicles.items():
            if vehicle_id not in self.counted_vehicles:
                # Vérifier si le véhicule a franchi la ligne
                if vehicle_data.get('crossed_line', False):
                    self.counted_vehicles.add(vehicle_id)
                    self.vehicle_count += 1
                    
                    # Ajouter à l'historique des comptages
                    self.count_history.append((current_time, self.vehicle_count))
        
        # Mettre à jour l'heure de la dernière mise à jour
        self.last_update_time = current_time
        
        return self.vehicle_count
    
    def draw_counter(self, frame):
        """
        Dessiner les informations du compteur sur l'image.
        
        Args:
            frame: Image sur laquelle dessiner
            
        Retourne:
            Image avec les informations du compteur dessinées
        """
        result_frame = frame.copy()
        
        # Dessiner la ligne de détection
        line_y = int(frame.shape[0] * self.detection_line_position)
        cv2.line(result_frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
        
        # Dessiner le texte du compteur
        counter_text = f"Comptage des véhicules : {self.vehicle_count}"
        cv2.putText(result_frame, counter_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame
    
    def get_count(self):
        """
        Obtenir le nombre actuel de véhicules.
        
        Retourne:
            Nombre actuel de véhicules
        """
        return self.vehicle_count
    
    def get_count_rate(self, time_window=60):
        """
        Calculer le taux de comptage des véhicules (véhicules par minute).
        
        Args:
            time_window: Fenêtre temporelle en secondes pour calculer le taux
            
        Retourne:
            Taux de comptage des véhicules (véhicules par minute)
        """
        # Filtrer l'historique des comptages pour ne prendre en compte que les entrées dans la fenêtre temporelle
        current_time = time.time()
        recent_counts = [(t, c) for t, c in self.count_history if current_time - t <= time_window]
        
        if not recent_counts:
            return 0
        
        # Calculer le taux basé sur le changement de comptage au fil du temps
        if len(recent_counts) >= 2:
            start_time, start_count = recent_counts[0]
            end_time, end_count = recent_counts[-1]
            
            time_diff = end_time - start_time
            count_diff = end_count - start_count
            
            if time_diff > 0:
                # Convertir en véhicules par minute
                return (count_diff / time_diff) * 60
        
        # Si nous n'avons pas assez de points de données, estimer en fonction du comptage total
        return (self.vehicle_count / time_window) * 60 if time_window > 0 else 0
    
    def reset(self):
        """
        Réinitialiser le compteur.
        """
        self.vehicle_count = 0
        self.counted_vehicles.clear()
        self.count_history.clear()
        self.last_update_time = time.time()
