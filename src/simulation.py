import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for fallback
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import List, Dict, Tuple, Optional
from drone_factory import create_drone
from drones.base_drone import BaseDrone
from environment import Environment, TerrainType


class Simulation:
    """
    A flexible drone simulation class that can handle multiple drones,
    different scenarios, and various collision avoidance strategies.
    """
    
    def __init__(self, 
                 name: str = "Drone Simulation",
                 duration: int = 20,
                 time_step: float = 0.1,
                 plot_bounds: Tuple[float, float, float, float] = (-10, 60, -10, 100),
                 environment: Optional[Environment] = None):
        """
        Initialize the simulation.
        
        Args:
            name: Name of the simulation
            duration: Number of simulation steps
            time_step: Time step between simulation frames
            plot_bounds: (x_min, x_max, y_min, y_max) for the plot area
            environment: Environment object with terrain features (optional)
        """
        self.name = name
        self.duration = duration
        self.time_step = time_step
        self.plot_bounds = plot_bounds
        
        # Environment setup
        if environment is None:
            self.environment = Environment(plot_bounds, f"{name} Environment")
        else:
            self.environment = environment
        
        # Simulation state
        self.drones: List[BaseDrone] = []
        self.goals: Dict[int, np.ndarray] = {}
        
        # Results tracking
        self.positions: Dict[int, List[List[float]]] = {}
        self.collisions: List[int] = []
        self.step_count = 0
        
        # Animation settings
        self.colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        
    def add_drone(self, 
                  drone_type: str, 
                  position: List[float], 
                  heading: float, 
                  goal: List[float]) -> int:
        """
        Add a drone to the simulation.
        
        Args:
            drone_type: Type of drone ("quadcopter", "fixedwing")
            position: Initial position [x, y, z]
            heading: Initial heading in degrees
            goal: Target position [x, y, z]
            
        Returns:
            int: Index of the added drone
        """
        
        # Check if starting position is safe
        if not self.environment.is_position_safe(position[0], position[1], position[2]):
            print(f"⚠️ Warning: Starting position {position} may not be safe for flight!")
        
        # Check if goal position is safe
        if not self.environment.is_position_safe(goal[0], goal[1], goal[2]):
            print(f"⚠️ Warning: Goal position {goal} may not be safe for flight!")
        
        drone = create_drone(drone_type, position, heading)
        drone_index = len(self.drones)
        
        self.drones.append(drone)
        self.goals[drone_index] = np.array(goal)
        self.positions[drone_index] = []
        
        return drone_index
    
    def remove_drone(self, drone_index):
        self.drones.remove(drone_index)

    def check_collision(self, drone_i: BaseDrone, drone_j: BaseDrone, expanded: int = 0) -> bool:
        """
        Check if two drones are colliding based on their 3D collision zones.
        
        Args:
            drone_i: First drone
            drone_j: Second drone
            expanded: Expanded collision zone 
            
        Returns:
            bool: True if drones are colliding
        """
        zi = drone_i.get_collision_zone()
        pos_j = np.array(drone_j.position)
        
        # Handle different collision zone types
        if len(zi) == 2:
            # Spherical collision zone (quadcopters)
            center, radius = zi
            center_array = np.array(center)
            
            # Calculate 3D distance
            distance = np.linalg.norm(center_array - pos_j)
            return distance <= (radius + expanded)
        
        elif len(zi) == 4:
            # 3D Box collision zone (fixed-wing)
            p1, p2, width, height = zi
            a, b, p = np.array(p1), np.array(p2), np.array(pos_j)
            
            # Check if point is within the 3D box defined by the line segment
            ab = b - a
            ap = p - a
            
            # Project point onto the line segment
            ab_dot = np.dot(ab, ab)
            if ab_dot == 0:
                return False
            
            t = np.dot(ap, ab) / ab_dot
            t = max(0, min(1, t))  # Clamp to [0, 1]
            
            # Find closest point on the line segment
            closest_on_line = a + t * ab
            
            # Calculate distance vector from line to point
            dist_vec = p - closest_on_line
            
            # Check if within width and height constraints
            horizontal_dist = np.linalg.norm(dist_vec[:2])  # x, y distance
            vertical_dist = abs(dist_vec[2]) if len(dist_vec) > 2 else 0  # z distance
            
            return (horizontal_dist <= (width / 2 + expanded) and 
                    vertical_dist <= (height / 2 + expanded))
        
        return False
    
    def detect_collisions(self, expanded: int = 0) -> List[Tuple[int, int]]:
        """
        Detect all current collisions between drones.
        
        Args:
            expanded: Expanded collision zone 
        
        Returns:
            List of tuples (i, j) representing colliding drone pairs
        """
        # Use list comprehension for clarity and efficiency
        # It iterates through all unique pairs of drones and checks for collisions.
        return [
            (i, j)
            for i in range(len(self.drones))
            for j in range(i + 1, len(self.drones))
            if self.check_collision(self.drones[i], self.drones[j], expanded)
        ]
    
    def get_other_drones(self, drone_index: int) -> List[BaseDrone]:
        """
        Get all other drones except the one at the given index.
        
        Args:
            drone_index: Index of the current drone
            
        Returns:
            List of other drones
        """
        return [self.drones[i] for i in range(len(self.drones)) if i != drone_index]
    
    def step(self) -> bool:
        """
        Execute one simulation step.
        
        The method detects actual collisions (expanded=0) for recording and 
        potential collisions (expanded > 0) for avoidance behavior.
        
        Uses a priority-based avoidance system: when two drones are on collision
        course, the drone with lower index continues on its path while the drone
        with higher index performs evasive maneuvers.
        
        Returns:
            bool: True if simulation should continue, False if finished
        """
        if self.step_count >= self.duration:
            return False
        
        # Detect actual collisions (expanded = 0)
        actual_collisions = self.detect_collisions(0)
        if actual_collisions:
            self.collisions.append(self.step_count)
        
        # Detect potential collisions for avoidance
        potential_collisions = self.detect_collisions(10)
        
        # Create avoidance mode mapping with priority system
        # Only one drone per collision pair should avoid, the other continues
        avoidance_mode = {i: False for i in range(len(self.drones))}
        for i, j in potential_collisions:
            # Priority system: lower index has priority and continues straight
            # Higher index drone performs avoidance maneuver
            if i < j:
                avoidance_mode[j] = True  # j avoids, i continues
            else:
                avoidance_mode[i] = True  # i avoids, j continues
        
        # Compute actions for each drone
        actions = []
        for i, drone in enumerate(self.drones):
            # Get other drones for collision avoidance
            other_drones = self.get_other_drones(i)
            
            # Get terrain effects at current position (3D)
            drone_pos = drone.position
            constraints = self.environment.get_flight_constraints_at_position(
                drone_pos[0], drone_pos[1], drone_pos[2]
            )
            weather_effects = self.environment.get_weather_effects()
            
            # Call the drone's own compute_action method
            action = drone.compute_action(
                goal=self.goals[i],
                avoid=avoidance_mode[i],
                other_drones=other_drones
            )
            
            # Apply terrain speed modifiers (3D)
            if hasattr(drone, 'max_speed') and isinstance(action, list):
                # For quadcopters with 3D velocity commands [vx, vy, vz]
                speed_mod = constraints['speed_modifier'] * weather_effects['speed_modifier']
                if len(action) == 3:
                    # Apply speed modifier to horizontal components only
                    action = [action[0] * speed_mod, action[1] * speed_mod, action[2]]
                    
            elif isinstance(action, list) and len(action) == 2:
                # For fixed-wing with 3D actions: [steering, climb_rate]
                steering, climb_rate = action
                if constraints['speed_modifier'] < 1.0:
                    steering *= constraints['speed_modifier']
                action = [steering, climb_rate]
            
            actions.append(action)
        
        # Move all drones
        for i, drone in enumerate(self.drones):
            drone.move(actions[i])
            self.positions[i].append(drone.position.copy())
        
        self.step_count += 1
        return True
    
    def run(self) -> Dict:
        """
        Run the complete simulation.
        
        Returns:
            Dict containing simulation results
        """
        print(f"🚁 Starting simulation: {self.name}")
        print(f"   Drones: {len(self.drones)}")
        print(f"   Duration: {self.duration} steps")
        
        while self.step():
            pass
        
        results = {
            "name": self.name,
            "steps": self.step_count,
            "collisions": len(self.collisions),
            "collision_steps": self.collisions,
            "positions": self.positions,
            "goals": self.goals,
            "drones": len(self.drones)
        }
        
        print(f"✅ Simulation completed!")
        print(f"   Total collisions: {len(self.collisions)}")
        
        return results
    
    def create_3d_animation(self, output_file: str = "simulation_3d.gif", interval: int = 100):
        """
        Create a 3D animated visualization of the simulation.
        
        Args:
            output_file: Path to save the animation
            interval: Time between frames in milliseconds
        """
        if not self.positions:
            print("❌ No simulation data available. Run simulation first.")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create line plots for 3D trajectories
        lines = []
        for i in range(len(self.drones)):
            line, = ax.plot([], [], [], color=self.colors[i % len(self.colors)], 
                           label=f"Drone {i}", linewidth=2)
            lines.append(line)
        
        # Add 3D goal markers
        for i, goal in self.goals.items():
            if len(goal) >= 3:
                ax.scatter(*goal, c='green', marker='x', s=100, label=f'Goal {i}')
            else:
                # If 2D goal provided, use default altitude
                ax.scatter(goal[0], goal[1], 50.0, c='green', marker='x', s=100, label=f'Goal {i}')
        
        # Set up 3D plot bounds
        x_min, x_max, y_min, y_max = self.plot_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Calculate altitude bounds from drone positions
        all_altitudes = []
        for drone_positions in self.positions.values():
            for pos in drone_positions:
                if len(pos) >= 3:
                    all_altitudes.append(pos[2])
        
        if all_altitudes:
            z_min = min(all_altitudes) - 10
            z_max = max(all_altitudes) + 10
        else:
            z_min, z_max = 0, 100
        
        ax.set_zlim(z_min, z_max)
        
        # Set labels
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_zlabel('Altitude (meters)')
        ax.legend(loc='upper right')
        
        # Add environment info to title
        env_info = f" | Terrain Zones: {len(self.environment.terrain_zones)}"
        weather = self.environment.weather_conditions
        if weather['wind_speed'] > 0:
            env_info += f" | Wind: {weather['wind_speed']:.1f}m/s"
        
        def update_3d(frame: int):
            # Update title with collision warning and environment info
            collision_warning = '⚠️ COLLISION!' if frame in self.collisions else ''
            ax.set_title(f"{self.name} (3D View) - Step {frame} {collision_warning}{env_info}")
            
            # Clear previous collision visualization
            # Note: 3D collision visualization could be added here
            
            # Update trajectory lines
            for i, line in enumerate(lines):
                if i < len(self.positions) and frame < len(self.positions[i]):
                    traj = np.array(self.positions[i][:frame+1])
                    if len(traj) > 0:
                        if traj.shape[1] >= 3:  # 3D positions
                            line.set_data_3d(traj[:, 0], traj[:, 1], traj[:, 2])
                        else:  # Handle legacy 2D positions by adding default altitude
                            default_alt = np.full(len(traj), 50.0)
                            line.set_data_3d(traj[:, 0], traj[:, 1], default_alt)
            
            # Add current drone positions as points
            for i, drone in enumerate(self.drones):
                if frame < len(self.positions[i]):
                    current_pos = self.positions[i][frame]
                    if len(current_pos) >= 3:
                        ax.scatter(current_pos[0], current_pos[1], current_pos[2], 
                                 c=self.colors[i % len(self.colors)], s=50, alpha=0.8)
                    else:  # Handle legacy 2D positions
                        ax.scatter(current_pos[0], current_pos[1], 50.0,
                                 c=self.colors[i % len(self.colors)], s=50, alpha=0.8)
        
        # Create animation
        total_frames = max(len(pos) for pos in self.positions.values()) if self.positions else 1
        ani = animation.FuncAnimation(fig, update_3d, frames=total_frames, 
                                    interval=interval, repeat=True)
        
        # Save animation
        ani.save(output_file, writer="pillow")
        print(f"📹 3D Animation saved as {output_file}")
        plt.close()
    
    def create_animation(self, output_file: str = "simulation_3d.gif", interval: int = 100):
        """
        Create a 3D animated visualization of the drone simulation.
        
        Args:
            output_file: Path to save the animation
            interval: Time between frames in milliseconds
        """
        if not self.positions:
            print("❌ No simulation data available. Run simulation first.")
            return
        
        print("🎬 Creating 3D visualization")
        self.create_3d_animation(output_file, interval)

    def create_interactive_3d_view(self, output_file: str = "simulation_interactive.html", auto_open: bool = True):
        """
        Create an interactive 3D visualization using Plotly with full navigation controls.
        
        This provides:
        - Interactive 3D navigation (rotate, zoom, pan)
        - Hover information for drones and goals
        - Play/pause animation controls
        - Camera angle controls
        - Terrain visualization
        - Real-time collision detection visualization
        
        Args:
            output_file: Path to save the HTML file
            auto_open: Whether to automatically open the visualization in browser
        """
        if not self.positions:
            print("❌ No simulation data available. Run simulation first.")
            return
        
        print("🚀 Creating interactive 3D visualization...")
        
        # Create the main 3D figure
        fig = go.Figure()
        
        # Define colors for drones
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Get bounds for the visualization
        all_positions = []
        for drone_positions in self.positions.values():
            all_positions.extend(drone_positions)
        
        if all_positions:
            all_pos_array = np.array(all_positions)
            x_range = [all_pos_array[:, 0].min() - 10, all_pos_array[:, 0].max() + 10]
            y_range = [all_pos_array[:, 1].min() - 10, all_pos_array[:, 1].max() + 10]
            z_range = [max(0, all_pos_array[:, 2].min() - 20), all_pos_array[:, 2].max() + 20]
        else:
            x_range, y_range, z_range = [-50, 100], [-50, 100], [0, 150]
        
        # Add terrain visualization
        self._add_terrain_to_plotly(fig)
        
        # Add drone trajectories
        for i, drone_positions in self.positions.items():
            if len(drone_positions) > 0:
                traj = np.array(drone_positions)
                color = colors[i % len(colors)]
                
                # Add trajectory line
                fig.add_trace(go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1], 
                    z=traj[:, 2],
                    mode='lines+markers',
                    line=dict(color=color, width=4),
                    marker=dict(size=3, color=color),
                    name=f'Drone {i} Path',
                    hovertemplate=f'<b>Drone {i}</b><br>' +
                                'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                '<extra></extra>'
                ))
                
                # Add start position marker
                fig.add_trace(go.Scatter3d(
                    x=[traj[0, 0]],
                    y=[traj[0, 1]],
                    z=[traj[0, 2]],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='diamond'),
                    name=f'Drone {i} Start',
                    hovertemplate=f'<b>Drone {i} Start</b><br>' +
                                'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                '<extra></extra>'
                ))
        
        # Add goal markers
        for i, goal in self.goals.items():
            if len(goal) >= 3:
                fig.add_trace(go.Scatter3d(
                    x=[goal[0]],
                    y=[goal[1]],
                    z=[goal[2]], 
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='yellow',
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    name=f'Goal {i}',
                    hovertemplate=f'<b>Goal {i}</b><br>' +
                                'Target: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                '<extra></extra>'
                ))
        
        # Add collision indicators
        if hasattr(self, 'collisions') and self.collisions:
            collision_points = []
            for step in self.collisions:
                for i, drone_positions in self.positions.items():
                    if step < len(drone_positions):
                        collision_points.append(drone_positions[step])
            
            if collision_points:
                collision_array = np.array(collision_points)
                fig.add_trace(go.Scatter3d(
                    x=collision_array[:, 0],
                    y=collision_array[:, 1],
                    z=collision_array[:, 2],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='x',
                        line=dict(color='darkred', width=3)
                    ),
                    name='Collisions',
                    hovertemplate='<b>⚠️ COLLISION!</b><br>' +
                                'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                '<extra></extra>'
                ))
        
        # Configure layout for optimal interactive experience
        fig.update_layout(
            title=dict(
                text=f"🚁 {self.name} - Interactive 3D Visualization",
                x=0.5,
                font=dict(size=20, color='darkblue')
            ),
            scene=dict(
                xaxis=dict(
                    title='X Position (m)',
                    range=x_range,
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='lightgray',
                    showbackground=True,
                    zerolinecolor='gray'
                ),
                yaxis=dict(
                    title='Y Position (m)', 
                    range=y_range,
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='lightgray',
                    showbackground=True,
                    zerolinecolor='gray'
                ),
                zaxis=dict(
                    title='Altitude (m)',
                    range=z_range,
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='lightgray',
                    showbackground=True,
                    zerolinecolor='gray'
                ),
                bgcolor='rgba(240,240,240,0.8)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube'
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            font=dict(size=12)
        )
        
        # Add annotations with simulation info
        total_distance = 0
        for drone_positions in self.positions.values():
            if len(drone_positions) > 1:
                traj = np.array(drone_positions)
                distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                total_distance += np.sum(distances)
        
        info_text = (
            f"📊 Simulation Statistics:<br>"
            f"• Total Steps: {self.step_count}<br>" 
            f"• Drones: {len(self.drones)}<br>"
            f"• Total Distance: {total_distance:.1f}m<br>"
            f"• Collisions: {len(self.collisions) if hasattr(self, 'collisions') else 0}"
        )
        
        fig.add_annotation(
            text=info_text,
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor='left', yanchor='bottom',
            showarrow=False,
            font=dict(size=11, color='darkblue'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1
        )
        
        # Save and optionally open the interactive visualization
        pyo.plot(fig, filename=output_file, auto_open=auto_open)
        print(f"🌐 Interactive 3D visualization saved as {output_file}")
        if auto_open:
            print("🚀 Opening in your default web browser...")
            print("💡 Use mouse to rotate, scroll to zoom, drag to pan!")
        
        return fig

    def _add_terrain_to_plotly(self, fig):
        """Add terrain zones to the Plotly figure."""
        if not hasattr(self.environment, 'terrain_zones'):
            return
            
        for zone in self.environment.terrain_zones:
            x_min, x_max, y_min, y_max = zone.bounds
            
            # Create terrain surface at ground level
            if zone.terrain_type.value in ['lake', 'river']:
                # Water features - blue with transparency
                color = 'rgba(100,150,255,0.3)'
                z_level = 0.5
            elif zone.terrain_type.value == 'forest':
                # Forest - green 
                color = 'rgba(50,150,50,0.4)'
                z_level = 1.0
            elif zone.terrain_type.value == 'mountain':
                # Mountains - gray/brown
                color = 'rgba(139,69,19,0.4)'
                z_level = 2.0
            elif zone.terrain_type.value == 'urban':
                # Urban - light gray
                color = 'rgba(180,180,180,0.4)'
                z_level = 1.5
            elif zone.terrain_type.value == 'no_fly_zone':
                # No-fly zone - red
                color = 'rgba(255,0,0,0.3)'
                z_level = 0.1
            else:
                # Open field - light green
                color = 'rgba(144,238,144,0.3)'
                z_level = 0.1
            
            # Add terrain as a surface
            x_terrain = [x_min, x_max, x_max, x_min, x_min]
            y_terrain = [y_min, y_min, y_max, y_max, y_min]
            z_terrain = [z_level] * 5
            
            fig.add_trace(go.Scatter3d(
                x=x_terrain,
                y=y_terrain,
                z=z_terrain,
                mode='lines',
                line=dict(color=color, width=0),
                fill='tonext' if zone.terrain_type.value != 'open_field' else None,
                name=f'{zone.terrain_type.value.title()} Zone',
                showlegend=True,
                hovertemplate=f'<b>{zone.terrain_type.value.title()} Zone</b><br>' +
                            f'Bounds: ({x_min:.1f}-{x_max:.1f}, {y_min:.1f}-{y_max:.1f})<br>' +
                            '<extra></extra>'
            ))

    def create_animated_interactive_view(self, output_file: str = "simulation_animated.html", 
                                       auto_open: bool = True, frame_duration: int = 200):
        """
        Create an animated interactive 3D visualization with frame-by-frame playback controls.
        
        Args:
            output_file: Path to save the HTML file
            auto_open: Whether to automatically open in browser
            frame_duration: Duration between frames in milliseconds
        """
        if not self.positions:
            print("❌ No simulation data available. Run simulation first.")
            return
            
        print("🎬 Creating animated interactive 3D visualization...")
        
        # Get the maximum number of frames
        max_frames = max(len(positions) for positions in self.positions.values())
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Create frames for animation
        frames = []
        for frame_num in range(max_frames):
            frame_data = []
            
            # Add terrain (static)
            self._add_terrain_frames(frame_data)
            
            # Add drone positions for this frame
            for i, drone_positions in self.positions.items():
                if frame_num < len(drone_positions):
                    pos = drone_positions[frame_num]
                    color = colors[i % len(colors)]
                    
                    # Current drone position
                    frame_data.append(go.Scatter3d(
                        x=[pos[0]],
                        y=[pos[1]],
                        z=[pos[2]],
                        mode='markers',
                        marker=dict(size=10, color=color, symbol='circle'),
                        name=f'Drone {i}',
                        hovertemplate=f'<b>Drone {i}</b><br>' +
                                    f'Step: {frame_num}<br>' +
                                    'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                    '<extra></extra>'
                    ))
                    
                    # Trail (path up to current frame)
                    if frame_num > 0:
                        trail = np.array(drone_positions[:frame_num+1])
                        frame_data.append(go.Scatter3d(
                            x=trail[:, 0],
                            y=trail[:, 1],
                            z=trail[:, 2],
                            mode='lines',
                            line=dict(color=color, width=3, dash='dot'),
                            name=f'Drone {i} Trail',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            # Check for collisions at this frame
            collision_warning = ''
            if hasattr(self, 'collisions') and frame_num in self.collisions:
                collision_warning = ' ⚠️ COLLISION!'
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(frame_num),
                layout=go.Layout(
                    title=f"🚁 {self.name} - Step {frame_num}{collision_warning}"
                )
            ))
        
        # Create the figure with the first frame
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # Add static goals that persist throughout the animation
        for i, goal in self.goals.items():
            if len(goal) >= 3:
                fig.add_trace(go.Scatter3d(
                    x=[goal[0]],
                    y=[goal[1]], 
                    z=[goal[2]],
                    mode='markers',
                    marker=dict(size=12, color='yellow', symbol='circle'),
                    name=f'Goal {i}',
                    hovertemplate=f'<b>Goal {i}</b><br>' +
                                'Target: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                '<extra></extra>'
                ))
        
        # Configure layout
        fig.update_layout(
            title=f"🚁 {self.name} - Animated Interactive 3D Visualization",
            scene=dict(
                xaxis=dict(title='X Position (m)', range=self.plot_bounds[:2]),
                yaxis=dict(title='Y Position (m)', range=self.plot_bounds[2:]),
                zaxis=dict(title='Altitude (m)', range=[0, 200]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='cube'
            ),
            updatemenus=[{
                'type': 'buttons',
                'direction': 'left',
                'showactive': False,
                'x': 0.1,
                'y': 0.02,
                'xanchor': 'right',
                'yanchor': 'bottom',
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': frame_duration, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f'Step {f.name}',
                        'method': 'animate'
                    } for f in frames
                ],
                'active': 0,
                'x': 0.1,
                'len': 0.8,
                'xanchor': 'left',
                'y': 0.02,
                'yanchor': 'bottom'
            }],
            width=1200,
            height=800
        )
        
        # Save and open
        pyo.plot(fig, filename=output_file, auto_open=auto_open)
        print(f"🎬 Animated interactive visualization saved as {output_file}")
        if auto_open:
            print("🚀 Opening in your default web browser...")
            print("💡 Use the play/pause buttons and slider to control the animation!")
        
        return fig

    def _add_terrain_frames(self, frame_data):
        """Helper method to add terrain data to animation frames."""
        # This would add terrain data consistently across frames
        # For now, keeping it simple, but could be expanded
        pass

