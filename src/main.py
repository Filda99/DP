from simulation import Simulation


def main():
    """Run the original simulation using the new Simulation class."""
    
    # Create simulation
    sim = Simulation(
        name="Original Drone Simulation",
        duration=60,
        plot_bounds=(-10, 60, -10, 100)
    )
    
    # Add the original two quadcopters
    sim.add_drone("quadcopter", [0., 0.], 0., [40., 40.])
    sim.add_drone("quadcopter", [10., 0.], 0., [0., 50.])
    
    # Run simulation (no need to set strategies - drones handle their own behavior)
    results = sim.run()
    
    # Create animation
    sim.create_animation("simulace.gif")
    
    print(f"\n📊 Simulation Results:")
    print(f"   Total steps: {results['steps']}")
    print(f"   Collisions detected: {results['collisions']}")
    print(f"   Collision steps: {results['collision_steps']}")


if __name__ == "__main__":
    main()
