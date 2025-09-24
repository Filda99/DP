"""
Modular PyBullet Drone Simulation Package

A comprehensive drone simulation system with:
- Multiple drone types (Quadcopter, Fixed-wing)
- Environmental features (Cities, forests, lakes, weather)
- Realistic physics simulation
- Comprehensive visualization and analysis

Main Components:
- drones: Quadcopter and FixedWing drone classes
- environment: Environmental features and weather system
- simulation: Complete simulation management
"""

from .drones import Quadcopter, FixedWing, BaseDrone
from .environment import Environment
from .simulation import Simulation

__version__ = "2.0.0"
__all__ = ['Quadcopter', 'FixedWing', 'BaseDrone', 'Environment', 'Simulation']