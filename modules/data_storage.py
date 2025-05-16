"""
Module de stockage des données pour sauvegarder les données de détection, de suivi et de vitesse des véhicules.
Ce module gère la sauvegarde des données dans des fichiers CSV et d'autres formats pour une analyse ultérieure.
"""

import os
import csv
import json
import pandas as pd
import datetime
import numpy as np

class DataStorage:
    def __init__(self, output_dir='data'):
        """
        Initialiser le module de stockage des données.
        
        Args:
            output_dir: Répertoire pour sauvegarder les fichiers de données
        """
        self.output_dir = output_dir
        self.vehicle_data = []  # Liste pour stocker les enregistrements des véhicules
        self.session_start_time = datetime.datetime.now()
        self.session_id = self.session_start_time.strftime("%Y %m %d_%H %M %S")
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
    
    def add_vehicle_record(self, vehicle_id, timestamp, bbox, speed=None, crossed_line=False):
        """
        Ajouter un enregistrement de détection de véhicule.
        
        Args:
            vehicle_id: ID du véhicule détecté
            timestamp: Horodatage de la détection
            bbox: Boîte englobante du véhicule (x, y, w, h)
            speed: Vitesse du véhicule en km/h (si disponible)
            crossed_line: Indique si le véhicule a franchi la ligne de détection
        """
        x, y, w, h = bbox
        
        record = {
            'session_id': self.session_id,
            'timestamp': timestamp,
            'vehicle_id': vehicle_id,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'speed': speed if speed is not None else -1,  # -1 indique que la vitesse n'est pas disponible
            'crossed_line': 1 if crossed_line else 0
        }
        
        self.vehicle_data.append(record)
    
    def save_to_csv(self, filename=None):
        """
        Sauvegarder les données des véhicules dans un fichier CSV.
        
        Args:
            filename: Nom du fichier CSV (si None, génère un nom par défaut)
            
        Retourne:
            Chemin vers le fichier CSV sauvegardé
        """
        if filename is None:
            filename = f"vehicle_data_{self.session_id}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convertir en DataFrame pour une manipulation plus facile
        df = pd.DataFrame(self.vehicle_data)
        
        # Sauvegarder dans un fichier CSV
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def save_summary(self, filename=None, vehicle_count=0, avg_speed=0):
        """
        Sauvegarder un résumé des données des véhicules.
        
        Args:
            filename: Nom du fichier de résumé (si None, génère un nom par défaut)
            vehicle_count: Nombre total de véhicules comptés
            avg_speed: Vitesse moyenne de tous les véhicules
            
        Retourne:
            Chemin vers le fichier de résumé sauvegardé
        """
        if filename is None:
            filename = f"summary_{self.session_id}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculer les statistiques si nous avons des données de véhicules
        if self.vehicle_data:
            df = pd.DataFrame(self.vehicle_data)
            
            # Filtrer les valeurs de vitesse invalides
            valid_speeds = df[df['speed'] >= 0]['speed']
            
            if not valid_speeds.empty:
                avg_speed = valid_speeds.mean()
                min_speed = valid_speeds.min()
                max_speed = valid_speeds.max()
                std_speed = valid_speeds.std()
            else:
                avg_speed = min_speed = max_speed = std_speed = 0
            
            # Compter les véhicules uniques
            if vehicle_count == 0:
                vehicle_count = df['vehicle_id'].nunique()
            
            # Compter les véhicules qui ont franchi la ligne
            crossed_count = df[df['crossed_line'] == 1]['vehicle_id'].nunique()
        else:
            min_speed = max_speed = std_speed = 0
            crossed_count = 0
        
        # Créer les données de résumé
        summary = {
            'session_id': self.session_id,
            'start_time': self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_vehicles': vehicle_count,
            'crossed_line': crossed_count,
            'avg_speed_kmh': round(avg_speed, 2),
            'min_speed_kmh': round(min_speed, 2),
            'max_speed_kmh': round(max_speed, 2),
            'std_speed_kmh': round(std_speed, 2)
        }
        
        # Sauvegarder dans un fichier CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)
        
        return filepath
    
    def save_speed_distribution(self, filename=None, bin_width=5):
        """
        Sauvegarder les données de distribution des vitesses.
        
        Args:
            filename: Nom du fichier de distribution (si None, génère un nom par défaut)
            bin_width: Largeur des bacs de vitesse en km/h
            
        Retourne:
            Chemin vers le fichier de distribution sauvegardé
        """
        if filename is None:
            filename = f"speed_distribution_{self.session_id}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculer la distribution des vitesses si nous avons des données de véhicules
        if self.vehicle_data:
            df = pd.DataFrame(self.vehicle_data)
            
            # Filtrer les valeurs de vitesse invalides et obtenir la dernière mesure de vitesse pour chaque véhicule
            valid_speeds = df[df['speed'] >= 0].sort_values('timestamp')
            last_speeds = valid_speeds.drop_duplicates('vehicle_id', keep='last')['speed']
            
            if not last_speeds.empty:
                # Créer des bacs pour l'histogramme
                max_speed = int(np.ceil(last_speeds.max()))
                bins = list(range(0, max_speed + bin_width, bin_width))
                
                # Calculer l'histogramme
                hist, bin_edges = np.histogram(last_speeds, bins=bins)
                
                # Créer les données de distribution
                distribution = []
                for i in range(len(hist)):
                    bin_start = bin_edges[i]
                    bin_end = bin_edges[i+1]
                    count = hist[i]
                    
                    distribution.append({
                        'bin_start': bin_start,
                        'bin_end': bin_end,
                        'bin_label': f"{bin_start}-{bin_end} km/h",
                        'count': count,
                        'percentage': round((count / len(last_speeds)) * 100, 2)
                    })
                
                # Sauvegarder dans un fichier CSV
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=distribution[0].keys())
                    writer.writeheader()
                    writer.writerows(distribution)
            else:
                # Créer un fichier de distribution vide
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['bin_start', 'bin_end', 'bin_label', 'count', 'percentage'])
        else:
            # Créer un fichier de distribution vide
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['bin_start', 'bin_end', 'bin_label', 'count', 'percentage'])
        
        return filepath
    
    def export_to_json(self, filename=None):
        """
        Exporter les données des véhicules dans un fichier JSON.
        
        Args:
            filename: Nom du fichier JSON (si None, génère un nom par défaut)
            
        Retourne:
            Chemin vers le fichier JSON sauvegardé
        """
        if filename is None:
            filename = f"vehicle_data_{self.session_id}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convertir les horodatages en chaînes de caractères pour la sérialisation JSON
        json_data = []
        for record in self.vehicle_data:
            json_record = record.copy()
            if isinstance(json_record['timestamp'], (datetime.datetime, pd.Timestamp)):
                json_record['timestamp'] = json_record['timestamp'].isoformat()
            json_data.append(json_record)
        
        # Sauvegarder dans un fichier JSON
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return filepath
    
    def clear_data(self):
        """
        Effacer toutes les données de véhicules stockées.
        """
        self.vehicle_data = []
        self.session_start_time = datetime.datetime.now()
        self.session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
