"""
Module de détection d'objets pour la détection de véhicules utilisant OpenCV.
Ce module utilise la soustraction de fond et la détection de contours pour identifier les véhicules.
"""

import cv2
import numpy as np
import os

class VehicleDetector:
    def __init__(self, min_area=500, detection_line_position=0.6):
        """
        Initialiser le détecteur de véhicules.
        
        Args:
            min_area (int): Aire minimale du contour à considérer comme un véhicule
            detection_line_position (float): Position de la ligne de détection (0-1)
        """
        # Initialiser le soustracteur de fond
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, 
            varThreshold=25, 
            detectShadows=True
        )
        
        self.min_area = min_area
        self.detection_line_position = detection_line_position
        
    def detect_vehicles(self, frame):
        """
        Détecter les véhicules dans l'image donnée.
        
        Args:
            frame: Image d'entrée de la vidéo
            
        Retourne:
            Liste des boîtes englobantes (x, y, w, h) pour les véhicules détectés
        """
        # Créer une copie de l'image
        result_frame = frame.copy()
        
        # Appliquer la soustraction de fond
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Suppression du bruit avec des opérations morphologiques
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Seuil pour obtenir un masque binaire
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Trouver les contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours en fonction de la surface et extraire les boîtes englobantes
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
                
                # Dessiner la boîte englobante sur l'image résultante
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Dessiner la ligne de détection
        line_y = int(frame.shape[0] * self.detection_line_position)
        cv2.line(result_frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
        
        return bounding_boxes, result_frame, line_y
    
    def save_model(self, model_path):
        """
        Sauvegarder les paramètres du soustracteur de fond (espace réservé pour la sauvegarde du modèle).
        Dans une implémentation réelle, vous pourriez sauvegarder les paramètres entraînés ici.
        
        Args:
            model_path: Chemin pour sauvegarder le modèle
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Dans une implémentation réelle, vous sauvegarderez ici les paramètres du modèle
        # Pour ce détecteur simple, nous n'avons pas de paramètres à sauvegarder
        with open(model_path, 'w') as f:
            f.write("VehicleDetector parameters:\n")
            f.write(f"min_area: {self.min_area}\n")
            f.write(f"detection_line_position: {self.detection_line_position}\n")
        
    def load_model(self, model_path):
        """
        Charger les paramètres du soustracteur de fond (espace réservé pour le chargement du modèle).
        Dans une implémentation réelle, vous pourriez charger les paramètres entraînés ici.
        
        Args:
            model_path: Chemin pour charger le modèle
        """
        # Dans une implémentation réelle, vous chargeriez ici les paramètres du modèle
        # Pour ce détecteur simple, nous n'avons pas de paramètres à charger
        try:
            with open(model_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "min_area:" in line:
                        self.min_area = int(line.split(":")[1].strip())
                    elif "detection_line_position:" in line:
                        self.detection_line_position = float(line.split(":")[1].strip())
        except FileNotFoundError:
            print(f"Le fichier du modèle {model_path} n'a pas été trouvé. Utilisation des paramètres par défaut.")
