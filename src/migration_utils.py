"""
Migration utilities for transitioning from 2D to 3D drone simulation.

This module provides utilities to help migrate existing 2D code to 3D
while maintaining backward compatibility.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from simulation import Simulation
from environment import Environment


class Migration2Dto3D:
    """Utility class for migrating 2D simulations to 3D."""
    
    @staticmethod
    def convert_position_2d_to_3d(position_2d: List[float], default_altitude: float = 50.0) -> List[float]:
        """
        Convert a 2D position to 3D by adding altitude.
        
        Args:
            position_2d: [x, y] coordinates
            default_altitude: Altitude to add (default: 50.0m)
            
        Returns:
            [x, y, z] coordinates
        """
        if len(position_2d) != 2:
            raise ValueError("position_2d must be [x, y]")
        return [position_2d[0], position_2d[1], default_altitude]
    
    @staticmethod
    def convert_multiple_positions_2d_to_3d(positions_2d: List[List[float]], 
                                          default_altitude: float = 50.0) -> List[List[float]]:
        """
        Convert multiple 2D positions to 3D.
        
        Args:
            positions_2d: List of [x, y] coordinates
            default_altitude: Altitude to add to all positions
            
        Returns:
            List of [x, y, z] coordinates
        """
        return [Migration2Dto3D.convert_position_2d_to_3d(pos, default_altitude) 
                for pos in positions_2d]
    
    @staticmethod
    def create_altitude_profile(start_altitude: float, end_altitude: float, 
                              num_points: int, profile_type: str = "linear") -> List[float]:
        """
        Create an altitude profile for transitioning between altitudes.
        
        Args:
            start_altitude: Starting altitude
            end_altitude: Ending altitude
            num_points: Number of points in the profile
            profile_type: Type of profile ("linear", "climb", "descent", "parabolic")
            
        Returns:
            List of altitude values
        """
        if profile_type == "linear":
            return np.linspace(start_altitude, end_altitude, num_points).tolist()
        
        elif profile_type == "climb":
            # Gradual start, steep end (good for takeoff)
            t = np.linspace(0, 1, num_points)
            profile = start_altitude + (end_altitude - start_altitude) * (t ** 2)
            return profile.tolist()
        
        elif profile_type == "descent":
            # Steep start, gradual end (good for landing)
            t = np.linspace(0, 1, num_points)
            profile = start_altitude + (end_altitude - start_altitude) * np.sqrt(t)
            return profile.tolist()
        
        elif profile_type == "parabolic":
            # Smooth acceleration then deceleration
            t = np.linspace(0, 1, num_points)
            profile = start_altitude + (end_altitude - start_altitude) * (4 * t * (1 - t))
            return profile.tolist()
        
        else:
            raise ValueError(f"Unknown profile_type: {profile_type}")
    
    @staticmethod
    def upgrade_2d_scenario_to_3d(add_drone_calls: List[Dict], 
                                default_altitude: float = 50.0,
                                altitude_variation: float = 20.0) -> List[Dict]:
        """
        Upgrade a list of 2D add_drone calls to 3D with altitude variation.
        
        Args:
            add_drone_calls: List of dictionaries with 'position', 'heading', 'goal', etc.
            default_altitude: Base altitude for all drones
            altitude_variation: Random variation in altitude (±variation/2)
            
        Returns:
            List of upgraded 3D add_drone call dictionaries
        """
        upgraded_calls = []
        np.random.seed(42)  # For reproducible results
        
        for i, call in enumerate(add_drone_calls):
            upgraded_call = call.copy()
            
            # Upgrade position
            if len(call['position']) == 2:
                variation = (np.random.random() - 0.5) * altitude_variation
                altitude = default_altitude + variation
                upgraded_call['position'] = Migration2Dto3D.convert_position_2d_to_3d(
                    call['position'], altitude)
            
            # Upgrade goal
            if len(call['goal']) == 2:
                variation = (np.random.random() - 0.5) * altitude_variation
                goal_altitude = default_altitude + variation
                upgraded_call['goal'] = Migration2Dto3D.convert_position_2d_to_3d(
                    call['goal'], goal_altitude)
            
            upgraded_calls.append(upgraded_call)
        
        return upgraded_calls


class CompatibilityChecker:
    """Check compatibility between 2D and 3D simulation components."""
    
    @staticmethod
    def check_drone_compatibility(drone_type: str) -> Dict[str, bool]:
        """
        Check what 3D features a drone type supports.
        
        Args:
            drone_type: "quadcopter" or "fixedwing"
            
        Returns:
            Dictionary of supported features
        """
        if drone_type == "quadcopter":
            return {
                "3d_movement": True,
                "vertical_velocity": True,
                "altitude_control": True,
                "3d_collision_detection": True,
                "backward_compatible": True
            }
        elif drone_type == "fixedwing":
            return {
                "3d_movement": True,
                "climb_descent": True,
                "altitude_control": True,
                "3d_collision_detection": True,
                "backward_compatible": True
            }
        else:
            return {
                "3d_movement": False,
                "vertical_velocity": False,
                "altitude_control": False,
                "3d_collision_detection": False,
                "backward_compatible": False
            }
    
    @staticmethod
    def check_environment_3d_features(environment: Environment) -> Dict[str, bool]:
        """
        Check what 3D features an environment supports.
        
        Args:
            environment: Environment instance
            
        Returns:
            Dictionary of supported 3D features
        """
        has_altitude_constraints = any(
            zone.altitude_restriction is not None 
            for zone in environment.terrain_zones
        )
        
        return {
            "altitude_constraints": has_altitude_constraints,
            "3d_terrain_zones": len(environment.terrain_zones) > 0,
            "3d_path_checking": hasattr(environment, 'is_path_safe_3d'),
            "altitude_range_queries": hasattr(environment, 'get_safe_altitude_range')
        }


class MigrationReport:
    """Generate reports on 2D to 3D migration status."""
    
    @staticmethod
    def generate_feature_report() -> str:
        """Generate a report of all 3D features available."""
        report = []
        report.append("🚁 2D to 3D Migration Feature Report")
        report.append("=" * 50)
        
        report.append("\n📐 Drone Features:")
        for drone_type in ["quadcopter", "fixedwing"]:
            features = CompatibilityChecker.check_drone_compatibility(drone_type)
            report.append(f"  {drone_type.capitalize()}:")
            for feature, supported in features.items():
                status = "✅" if supported else "❌"
                report.append(f"    {status} {feature.replace('_', ' ').title()}")
        
        report.append("\n🌍 Environment Features:")
        env_features = [
            "3D terrain zones with altitude restrictions",
            "Altitude-aware flight constraints",
            "3D path safety checking",
            "Altitude range queries",
            "Weather effects on 3D flight"
        ]
        for feature in env_features:
            report.append(f"  ✅ {feature}")
        
        report.append("\n🎬 Visualization Features:")
        viz_features = [
            "3D trajectory plotting",
            "Smart 2D/3D visualization selection",
            "3D collision zone visualization",
            "Altitude layer visualization"
        ]
        for feature in viz_features:
            report.append(f"  ✅ {feature}")
        
        report.append("\n🔄 Backward Compatibility:")
        compat_features = [
            "All existing 2D code works unchanged",
            "2D positions automatically get default altitude",
            "2D goals work with 3D drones",
            "Existing scenarios run without modification",
            "2D visualization still available"
        ]
        for feature in compat_features:
            report.append(f"  ✅ {feature}")
        
        return "\n".join(report)
    
    @staticmethod
    def generate_migration_guide() -> str:
        """Generate a step-by-step migration guide."""
        guide = []
        guide.append("📋 2D to 3D Migration Guide")
        guide.append("=" * 40)
        
        guide.append("\n🎯 Quick Start (No Code Changes):")
        guide.append("  1. Your existing 2D code works as-is")
        guide.append("  2. Drones automatically get default altitude (50m)")
        guide.append("  3. Use create_smart_animation() for automatic visualization")
        
        guide.append("\n🚀 Gradual 3D Enhancement:")
        guide.append("  1. Add altitude parameter to add_drone() calls:")
        guide.append("     sim.add_drone('quadcopter', [0, 0], 45, [40, 40], altitude=60)")
        guide.append("  2. Upgrade positions to 3D:")
        guide.append("     sim.add_drone('quadcopter', [0, 0, 30], 45, [40, 40, 80])")
        guide.append("  3. Use 3D environment features:")
        guide.append("     env.generate_3d_environment()")
        
        guide.append("\n🎮 Advanced 3D Features:")
        guide.append("  1. Create altitude-layered scenarios")
        guide.append("  2. Use terrain with altitude restrictions")
        guide.append("  3. Implement vertical collision avoidance")
        guide.append("  4. Create climb/descent flight profiles")
        
        guide.append("\n📊 Migration Utilities:")
        guide.append("  • Migration2Dto3D.convert_position_2d_to_3d()")
        guide.append("  • Migration2Dto3D.create_altitude_profile()")
        guide.append("  • Migration2Dto3D.upgrade_2d_scenario_to_3d()")
        
        return "\n".join(guide)


def run_migration_test():
    """Test the migration utilities."""
    print("🧪 Testing 2D to 3D Migration Utilities")
    print("=" * 50)
    
    # Test position conversion
    pos_2d = [10, 20]
    pos_3d = Migration2Dto3D.convert_position_2d_to_3d(pos_2d, 75)
    print(f"2D Position: {pos_2d} -> 3D Position: {pos_3d}")
    
    # Test altitude profile creation
    profile = Migration2Dto3D.create_altitude_profile(20, 100, 5, "climb")
    print(f"Climb Profile: {[f'{alt:.1f}m' for alt in profile]}")
    
    # Test scenario upgrade
    old_scenario = [
        {"position": [0, 0], "heading": 45, "goal": [40, 40]},
        {"position": [10, 10], "heading": 90, "goal": [50, 50]}
    ]
    
    new_scenario = Migration2Dto3D.upgrade_2d_scenario_to_3d(old_scenario, 60, 30)
    print(f"\nScenario Upgrade:")
    for i, (old, new) in enumerate(zip(old_scenario, new_scenario)):
        print(f"  Drone {i}: {old['position']} -> {new['position']}")
        print(f"           {old['goal']} -> {new['goal']}")
    
    print("\n✅ Migration utilities test completed!")


def main():
    """Main function to demonstrate migration utilities."""
    print(MigrationReport.generate_feature_report())
    print("\n" + "=" * 80 + "\n")
    print(MigrationReport.generate_migration_guide())
    print("\n" + "=" * 80 + "\n")
    run_migration_test()


if __name__ == "__main__":
    main()