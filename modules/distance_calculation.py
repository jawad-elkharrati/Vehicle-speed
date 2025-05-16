"""
Module de calcul de la distance pour mesurer les distances réelles à partir des coordonnées en pixels.
Ce module gère la conversion entre les distances en pixels et les mesures réelles.
"""

import numpy as np
import cv2

class DistanceCalculator:
    def __init__(self, reference_width_pixels=None, reference_width_meters=None):
        """
        Initialiser le calculateur de distance.
        
        Args:
            reference_width_pixels: Largeur d'un objet de référence en pixels
            reference_width_meters: Largeur du même objet de référence en mètres
        """
        self.reference_width_pixels = reference_width_pixels
        self.reference_width_meters = reference_width_meters
        self.pixels_per_meter = None
        
        if reference_width_pixels is not None and reference_width_meters is not None:
            self.pixels_per_meter = reference_width_pixels / reference_width_meters
    
    def set_reference(self, reference_width_pixels, reference_width_meters):
        """
        Définir les mesures de référence pour la conversion des pixels en mètres.
        
        Args:
            reference_width_pixels: Largeur d'un objet de référence en pixels
            reference_width_meters: Largeur du même objet de référence en mètres
        """
        self.reference_width_pixels = reference_width_pixels
        self.reference_width_meters = reference_width_meters
        self.pixels_per_meter = reference_width_pixels / reference_width_meters
    
    def pixel_to_meter(self, pixel_distance):
        """
        Convertir une distance en pixels en mètres.
        
        Args:
            pixel_distance: Distance en pixels
            
        Retourne:
            Distance en mètres
        """
        if self.pixels_per_meter is None:
            raise ValueError("Les mesures de référence ne sont pas définies. Appelez set_reference() d'abord.")
        
        return pixel_distance / self.pixels_per_meter
    
    def meter_to_pixel(self, meter_distance):
        """
        Convertir une distance en mètres en pixels.
        
        Args:
            meter_distance: Distance en mètres
            
        Retourne:
            Distance en pixels
        """
        if self.pixels_per_meter is None:
            raise ValueError("Les mesures de référence ne sont pas définies. Appelez set_reference() d'abord.")
        
        return meter_distance * self.pixels_per_meter
    
    def calculate_distance(self, point1, point2):
        """
        Calculer la distance entre deux points en mètres.
        
        Args:
            point1: Premier point (x, y) en pixels
            point2: Deuxième point (x, y) en pixels
            
        Retourne:
            Distance en mètres
        """
        # Calculer la distance euclidienne en pixels
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        # Convertir en mètres
        return self.pixel_to_meter(pixel_distance)
    
    def calibrate_from_lane_width(self, frame, lane_width_meters=3.5):
        """
        Calibrer le calculateur de distance en utilisant une largeur de voie standard.
        Il s'agit d'une méthode de calibration simplifiée qui suppose une largeur de voie standard.
        
        Args:
            frame: Image vidéo pour la calibration
            lane_width_meters: Largeur d'une voie standard en mètres (par défaut 3.5m)
            
        Retourne:
            True si la calibration a réussi, False sinon
        """
        # Ceci est un espace réservé pour une méthode de calibration plus sophistiquée
        # Dans une implémentation réelle, vous pourriez utiliser la détection de voies ou demander à l'utilisateur de marquer les limites des voies
        
        # Pour simplifier, on suppose que la largeur de la voie est approximativement 1/3 de la largeur de l'image
        frame_width = frame.shape[1]
        estimated_lane_width_pixels = frame_width / 3
        
        self.set_reference(estimated_lane_width_pixels, lane_width_meters)
        return True
    
    def draw_reference_line(self, frame, start_point=None, end_point=None, color=(0, 255, 255), thickness=2):
        """
        Dessiner une ligne de référence sur l'image pour visualiser la distance de calibration.
        
        Args:
            frame: Image sur laquelle dessiner
            start_point: Point de départ de la ligne (x, y)
            end_point: Point d'arrivée de la ligne (x, y)
            color: Couleur de la ligne (B, G, R)
            thickness: Épaisseur de la ligne
            
        Retourne:
            Image avec la ligne de référence dessinée
        """
        result_frame = frame.copy()
        
        if start_point is None or end_point is None:
            # Si les points ne sont pas fournis, dessiner une ligne horizontale dans le tiers inférieur de l'image
            height, width = frame.shape[:2]
            y_position = int(height * 2/3)
            start_point = (int(width/3), y_position)
            end_point = (int(2*width/3), y_position)
        
        cv2.line(result_frame, start_point, end_point, color, thickness)
        
        # Calculer et afficher la distance
        if self.pixels_per_meter is not None:
            pixel_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            meter_distance = self.pixel_to_meter(pixel_distance)
            
            # Afficher la distance sur l'image
            mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2 - 10)
            cv2.putText(result_frame, f"{meter_distance:.2f} m", mid_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result_frame
