"""
Fichier __init__.py pour rendre le répertoire des modules un véritable package Python.
"""

# Importer tous les modules pour un accès plus facile
from .object_detection import VehicleDetector
from .vehicle_tracking import VehicleTracker
from .distance_calculation import DistanceCalculator
from .speed_calculation import SpeedCalculator
from .vehicle_counting import VehicleCounter
from .data_storage import DataStorage

__all__ = [
    'VehicleDetector',
    'VehicleTracker',
    'DistanceCalculator',
    'SpeedCalculator',
    'VehicleCounter',
    'DataStorage'
]
