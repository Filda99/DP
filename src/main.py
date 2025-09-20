from simulation import Simulation


def main():
    """Run the original simulation using the new 3D Simulation class."""
    
    # Create simulation
    sim = Simulation(
        name="Original Drone Simulation (3D)",
        duration=60,
        plot_bounds=(-10, 60, -10, 100)
    )
    
    # Add the original two quadcopters in 3D
    sim.add_drone("quadcopter", [0., 0., 50.], 0., [40., 40., 50.])
    sim.add_drone("quadcopter", [10., 0., 60.], 0., [0., 50., 60.])
    
    # Run simulation (no need to set strategies - drones handle their own behavior)
    results = sim.run()
    
    # Create 3D animation
    sim.create_3d_animation("simulace_3d.gif")
    
    print(f"\n📊 Simulation Results:")
    print(f"   Total steps: {results['steps']}")
    print(f"   Collisions detected: {results['collisions']}")
    print(f"   Collision steps: {results['collision_steps']}")


if __name__ == "__main__":
    main()
