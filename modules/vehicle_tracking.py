"""
Module de suivi des véhicules pour suivre les véhicules détectés à travers plusieurs frames.
Ce module implémente un algorithme simple de suivi basé sur le recouvrement des boîtes englobantes.
"""

import numpy as np

class VehicleTracker:
    def __init__(self, max_disappeared=10, min_overlap_ratio=0.3):
        """
        Initialiser le suiveur de véhicules.
        
        Args:
            max_disappeared (int): Nombre maximum de frames pendant lesquelles un véhicule peut disparaître avant d'être supprimé
            min_overlap_ratio (float): Ratio de recouvrement minimal pour considérer que c'est le même véhicule
        """
        self.next_vehicle_id = 0
        self.vehicles = {}  # Dictionnaire pour stocker les véhicules actifs {id: {'bbox': (x,y,w,h), 'disappeared': count}}
        self.max_disappeared = max_disappeared
        self.min_overlap_ratio = min_overlap_ratio
        self.crossed_vehicles = set()  # Ensemble des IDs des véhicules ayant franchi la ligne de détection
        
    def _calculate_overlap(self, bbox1, bbox2):
        """
        Calculer le ratio de recouvrement entre deux boîtes englobantes.
        
        Args:
            bbox1: Première boîte englobante (x, y, w, h)
            bbox2: Deuxième boîte englobante (x, y, w, h)
            
        Retourne:
            Ratio de recouvrement (aire de l'intersection / aire de l'union)
        """
        # Extraire les coordonnées
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculer les coordonnées d'intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Vérifier s'il y a une intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculer les surfaces
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculer le ratio de recouvrement
        overlap_ratio = intersection_area / float(union_area)
        
        return overlap_ratio
    
    def update(self, bounding_boxes, detection_line_y):
        """
        Mettre à jour le suiveur avec de nouvelles boîtes englobantes.
        
        Args:
            bounding_boxes: Liste des boîtes englobantes (x, y, w, h) pour les véhicules détectés
            detection_line_y: Coordonnée y de la ligne de détection
            
        Retourne:
            Dictionnaire des véhicules suivis avec leurs IDs et boîtes englobantes
        """
        # Si aucune boîte englobante, marquer tous les véhicules existants comme disparus
        if len(bounding_boxes) == 0:
            for vehicle_id in list(self.vehicles.keys()):
                self.vehicles[vehicle_id]['disappeared'] += 1
                
                # Supprimer le véhicule s'il a disparu trop longtemps
                if self.vehicles[vehicle_id]['disappeared'] > self.max_disappeared:
                    del self.vehicles[vehicle_id]
                    
            return self.vehicles
        
        # Si aucun véhicule existant, enregistrer toutes les boîtes englobantes comme nouveaux véhicules
        if len(self.vehicles) == 0:
            for bbox in bounding_boxes:
                self.register(bbox, detection_line_y)
        else:
            # Essayer de faire correspondre les véhicules existants avec les nouvelles boîtes englobantes
            vehicle_ids = list(self.vehicles.keys())
            vehicle_bboxes = [self.vehicles[vehicle_id]['bbox'] for vehicle_id in vehicle_ids]
            
            # Suivre quels véhicules et quelles boîtes englobantes ont été appariés
            used_vehicles = set()
            used_bboxes = set()
            
            # Pour chaque véhicule, trouver la meilleure boîte englobante correspondante
            for i, vehicle_id in enumerate(vehicle_ids):
                vehicle_bbox = vehicle_bboxes[i]
                
                max_overlap = 0
                max_overlap_idx = -1
                
                for j, bbox in enumerate(bounding_boxes):
                    if j in used_bboxes:
                        continue
                    
                    overlap = self._calculate_overlap(vehicle_bbox, bbox)
                    
                    if overlap > max_overlap and overlap > self.min_overlap_ratio:
                        max_overlap = overlap
                        max_overlap_idx = j
                
                # Si une correspondance a été trouvée, mettre à jour le véhicule
                if max_overlap_idx != -1:
                    self.vehicles[vehicle_id]['bbox'] = bounding_boxes[max_overlap_idx]
                    self.vehicles[vehicle_id]['disappeared'] = 0
                    used_vehicles.add(vehicle_id)
                    used_bboxes.add(max_overlap_idx)
                    
                    # Vérifier si le véhicule a franchi la ligne de détection
                    self._check_line_crossing(vehicle_id, bounding_boxes[max_overlap_idx], detection_line_y)
            
            # Marquer les véhicules non appariés comme disparus
            for vehicle_id in vehicle_ids:
                if vehicle_id not in used_vehicles:
                    self.vehicles[vehicle_id]['disappeared'] += 1
                    
                    # Supprimer le véhicule s'il a disparu trop longtemps
                    if self.vehicles[vehicle_id]['disappeared'] > self.max_disappeared:
                        del self.vehicles[vehicle_id]
            
            # Enregistrer les boîtes englobantes non appariées comme nouveaux véhicules
            for i, bbox in enumerate(bounding_boxes):
                if i not in used_bboxes:
                    self.register(bbox, detection_line_y)
        
        return self.vehicles
    
    def register(self, bbox, detection_line_y):
        """
        Enregistrer un nouveau véhicule.
        
        Args:
            bbox: Boîte englobante (x, y, w, h) du nouveau véhicule
            detection_line_y: Coordonnée y de la ligne de détection
        """
        self.vehicles[self.next_vehicle_id] = {
            'bbox': bbox,
            'disappeared': 0,
            'crossed_line': False,
            'first_seen': {'bbox': bbox, 'crossed_line': False}
        }
        
        # Vérifier si le véhicule a franchi la ligne de détection
        self._check_line_crossing(self.next_vehicle_id, bbox, detection_line_y)
        
        self.next_vehicle_id += 1
    
    def _check_line_crossing(self, vehicle_id, bbox, detection_line_y):
        """
        Vérifier si un véhicule a franchi la ligne de détection.
        
        Args:
            vehicle_id: ID du véhicule
            bbox: Boîte englobante actuelle (x, y, w, h) du véhicule
            detection_line_y: Coordonnée y de la ligne de détection
        """
        x, y, w, h = bbox
        vehicle_bottom = y + h
        
        # Si le bas du véhicule franchit la ligne de détection
        if vehicle_bottom >= detection_line_y and not self.vehicles[vehicle_id]['crossed_line']:
            self.vehicles[vehicle_id]['crossed_line'] = True
            self.crossed_vehicles.add(vehicle_id)
        
    def get_crossed_count(self):
        """
        Obtenir le nombre de véhicules ayant franchi la ligne de détection.
        
        Retourne:
            Nombre de véhicules ayant franchi la ligne
        """
        return len(self.crossed_vehicles)
    
    def get_active_vehicles(self):
        """
        Obtenir les véhicules actuellement actifs.
        
        Retourne:
            Dictionnaire des véhicules actifs avec leurs IDs et boîtes englobantes
        """
        return {vehicle_id: vehicle_data for vehicle_id, vehicle_data in self.vehicles.items() 
                if vehicle_data['disappeared'] == 0}
