#!/usr/bin/env python3
"""
Analyze Simulation Log

Load a simulation log file and generate visualizations/analysis.
This script is separate from the main simulation to keep simulation runs fast.
"""

import json
import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_log(log_file):
    """Load simulation log from JSON file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def print_summary(log_data):
    """Print summary of simulation."""
    metadata = log_data['metadata']
    events = log_data['events']
    
    print("=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Total time: {metadata['total_time']:.2f}s")
    print(f"Timestep: {metadata['timestep']:.4f}s")
    print(f"Active drones: {', '.join(metadata['drones'])}")
    print(f"Destroyed drones: {', '.join(metadata['destroyed_drones']) if metadata['destroyed_drones'] else 'None'}")
    print()
    
    # Count event types
    event_types = {}
    for event in events:
        event_type = event['event']
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print("Events:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    print("=" * 70)


def plot_fire_progression(log_data, output_file='fire_progression.png'):
    """Plot fire progression over time."""
    fire_states = log_data['simulation_log']['fire_states']
    times = log_data['simulation_log']['times']
    
    if not fire_states:
        print("No fire data to plot")
        return
    
    # Extract fire statistics
    burning_cells = []
    for state in fire_states:
        if state and 'fire_stats' in state:
            burning_cells.append(state['fire_stats'].get('burning_cells', 0))
        else:
            burning_cells.append(0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times[:len(burning_cells)], burning_cells, 'r-', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Burning Cells', fontsize=12)
    plt.title('Fire Progression Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved fire progression plot: {output_file}")
    plt.close()


def plot_drone_trajectories(log_data, output_file='drone_trajectories.png'):
    """Plot drone trajectories in 2D."""
    drone_logs = log_data['simulation_log']['drones']
    
    if not drone_logs:
        print("No drone data to plot")
        return
    
    plt.figure(figsize=(12, 12))
    
    for drone_name, drone_data in drone_logs.items():
        positions = np.array(drone_data['positions'])
        if len(positions) > 0:
            plt.plot(positions[:, 0], positions[:, 1], '-', label=drone_name, alpha=0.7, linewidth=2)
            plt.plot(positions[0, 0], positions[0, 1], 'o', markersize=10, label=f'{drone_name} start')
            plt.plot(positions[-1, 0], positions[-1, 1], 'x', markersize=10, label=f'{drone_name} end')
    
    # Mark collisions
    collisions = log_data['simulation_log']['collisions']
    for collision in collisions:
        pos = collision['position']
        plt.plot(pos[0], pos[1], 'r*', markersize=15, markeredgecolor='black', markeredgewidth=1)
    
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title('Drone Trajectories (Top View)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved drone trajectories plot: {output_file}")
    plt.close()


def plot_collision_timeline(log_data, output_file='collision_timeline.png'):
    """Plot collision events over time."""
    events = log_data['events']
    
    collision_events = [e for e in events if e['event'] in ['collision', 'ground_crash', 'drone_collision']]
    
    if not collision_events:
        print("No collision events to plot")
        return
    
    times = [e['time'] for e in collision_events]
    event_types = [e['event'] for e in collision_events]
    drones = [e['data']['drone'] for e in collision_events]
    
    plt.figure(figsize=(14, 6))
    
    # Color map for event types
    colors = {
        'collision': 'red',
        'ground_crash': 'orange',
        'drone_collision': 'purple'
    }
    
    for i, (t, event_type, drone) in enumerate(zip(times, event_types, drones)):
        plt.scatter(t, i, c=colors.get(event_type, 'gray'), s=200, alpha=0.7, edgecolors='black', linewidths=2)
        plt.text(t, i, f' {drone}', va='center', fontsize=10)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Collision Event Index', fontsize=12)
    plt.title('Collision Timeline', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Legend
    for event_type, color in colors.items():
        plt.scatter([], [], c=color, s=100, label=event_type, edgecolors='black', linewidths=1)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved collision timeline plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze simulation log and generate visualizations')
    parser.add_argument('log_file', help='Path to simulation log JSON file')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory for plots')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary, no plots')
    
    args = parser.parse_args()
    
    # Load log
    print(f"📂 Loading log: {args.log_file}")
    log_data = load_log(args.log_file)
    
    # Print summary
    print_summary(log_data)
    
    if args.summary_only:
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📊 Generating visualizations in: {args.output_dir}")
    
    # Generate plots
    plot_fire_progression(log_data, os.path.join(args.output_dir, 'fire_progression.png'))
    plot_drone_trajectories(log_data, os.path.join(args.output_dir, 'drone_trajectories.png'))
    plot_collision_timeline(log_data, os.path.join(args.output_dir, 'collision_timeline.png'))
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
