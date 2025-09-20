from simulation import Simulation


def main():
    """Run the simulation with interactive 3D visualization."""
    
    # Create simulation
    sim = Simulation(
        name="Interactive Drone Simulation (3D)",
        duration=60,
        plot_bounds=(-10, 60, -10, 100)
    )
    
    # Add the original two quadcopters in 3D
    sim.add_drone("quadcopter", [0., 0., 50.], 0., [40., 40., 70.])
    sim.add_drone("quadcopter", [10., 0., 60.], 0., [0., 50., 40.])
    
    # Run simulation
    results = sim.run()
    
    print(f"\n📊 Simulation Results:")
    print(f"   Total steps: {results['steps']}")
    print(f"   Collisions detected: {results['collisions']}")
    print(f"   Collision steps: {results['collision_steps']}")
    
    print(f"\n🚀 Creating Interactive Animated Visualization...")
    
    # Create only the animated interactive view with goals
    print("\n🎬 Creating animated interactive view with playback controls...")
    sim.create_animated_interactive_view("animated_interactive.html", auto_open=True)
    
    print(f"\n✅ Interactive animation created!")
    print(f"📂 File created:")
    print(f"   • animated_interactive.html - Animated with playback controls")
    print(f"\n💡 The interactive animation provides:")
    print(f"   🖱️  Mouse controls: rotate, zoom, pan")
    print(f"   🎯 Goals: See where each drone is flying to")
    print(f"   ℹ️  Hover for detailed information")
    print(f"   🎮 Animation controls (play/pause/slider)")
    print(f"   📊 Built-in statistics and collision indicators")


if __name__ == "__main__":
    main()
