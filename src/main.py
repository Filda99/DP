import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from drone_factory import create_drone
from drones.base_drone import BaseDrone

# === Drony a cíle ===
drones = [
    create_drone("quadcopter", [0., 0.], 0.),
    create_drone("quadcopter", [10., 0.], 0.)
    # create_drone("fixedwing", [50., 0.], 90.)
]
for d in drones:
    d.dt = 0.5

goals = {
    0: np.array([40., 40.]),
    1: np.array([0., 50.])
}

positions = {i: [] for i in range(len(drones))}
collisions = []
avoidance_mode = [False for _ in drones]

# === Akční strategie ===
def quad_action(drone: BaseDrone, goal: np.ndarray, avoid: bool = False) -> list:
    """
    Computes movement action for a quadcopter drone.
    
    Args:
        drone: The drone object with current position
        goal: Target position coordinates 
        avoid: If True, performs evasive maneuver
    
    Returns:
        List of [x_velocity, y_velocity] for movement
    """
    # If avoiding collision, move diagonally left-up
    if avoid:
        return [-1.0, 1.0]
    
    # Calculate direction vector from current position to goal
    vec = np.array(goal) - np.array(drone.position)
    norm = np.linalg.norm(vec)
    
    # If very close to goal (within 1 unit), stop moving
    if norm < 1.0:
        return [0, 0]
    
    # Return normalized direction vector (unit vector pointing toward goal)
    return (vec / norm).tolist()

def fixedwing_action(drone: BaseDrone, goal: np.ndarray, avoid: bool = False) -> float:
    """
    Computes steering action for a fixed-wing drone.
    
    Args:
        drone: The drone object with current position and heading
        goal: Target position coordinates
        avoid: If True, performs evasive turn
    
    Returns:
        Steering angle in degrees (positive = right turn, negative = left turn)
    """
    import math
    
    # If avoiding collision, perform sharp left turn
    if avoid:
        return -15
    
    # Get current position and calculate vector to goal
    pos = np.array(drone.position)
    vec = np.array(goal) - pos
    
    # Calculate target angle (direction to goal) in degrees
    target_angle = math.degrees(math.atan2(vec[1], vec[0]))
    
    # Calculate angular difference between current heading and target
    delta = (target_angle - drone.heading + 360) % 360
    
    # Normalize angle difference to range [-180, 180]
    if delta > 180:
        delta -= 360
    
    # Limit steering to maximum ±15 degrees per step
    delta = max(min(delta, 15), -15)
    return delta


# === Kolizní kontrola ===
def check_collision(di: BaseDrone, dj: BaseDrone) -> bool:
    """
    Checks if two drones are colliding based on their collision zones.
    
    Args:
        di: First drone object
        dj: Second drone object
    
    Returns:
        bool: True if drones are colliding, False otherwise
    """
    # Get collision zone of first drone and position of second drone
    zi = di.get_collision_zone()
    zj = dj.position
    
    # Check circular collision zone (format: [center, radius]) for quadcopters
    if len(zi) == 2:
        center, radius = zi
        # Calculate distance between drone center and other drone's position
        return np.linalg.norm(np.array(center) - np.array(zj)) <= radius
    
    # Check rectangular collision zone (format: [point1, point2, width]) for fixed-wing
    # where point1 and point2 are the ends of the rectangle's long side
    # and width is the short side width
    elif len(zi) == 3:
        p1, p2, width = zi
        a, b, p = np.array(p1), np.array(p2), np.array(zj)
        
        # Calculate vectors for projection
        ab, ap = b - a, p - a
        
        # Project point onto line segment
        proj = np.dot(ap, ab) / np.dot(ab, ab)
        closest = a + proj * ab
        
        # Calculate distance from point to closest point on line
        dist = np.linalg.norm(closest - p)
        
        # Check if point is within rectangle bounds
        return 0 <= proj <= 1 and dist <= width / 2
    
    # Unknown collision zone format
    return False


# === SIMULACE ===
for step in range(60):
    colliding = check_collision(drones[0], drones[1])
    if colliding:
        collisions.append(step)
        avoidance_mode = [True, True]
    else:
        avoidance_mode = [False, False]

    acts = [
        quad_action(drones[0], goals[0], avoidance_mode[0]),
        fixedwing_action(drones[1], goals[1], avoidance_mode[1])
    ]

    for i, drone in enumerate(drones):
        drone.move(acts[i])
        positions[i].append(drone.position.copy())


# === ANIMACE ===
fig, ax = plt.subplots()
colors = ["blue", "red"]
labels = ["Quad", "FixedWing"]
scatters = [ax.plot([], [], color=colors[i], label=labels[i])[0] for i in range(len(drones))]
goal_marks = [ax.scatter(*goals[i], c='green', marker='x', s=100) for i in range(len(drones))]

ax.set_xlim(-10, 60)
ax.set_ylim(-10, 100)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()

def update(frame: int) -> None:
    # Set window title with current frame and collision warning
    ax.set_title(f"Krok {frame} {'⚠️ Kolize!' if frame in collisions else ''}")
    
    # Remove previous collision zone patches from the plot
    for patch in reversed(ax.patches):
        patch.remove()

    # Update trajectory lines for each drone
    for i in range(len(drones)):
        # Get trajectory data up to current frame
        traj = np.array(positions[i][:frame+1])
        if len(traj) > 1:
            # Update the line plot with new trajectory data
            scatters[i].set_data(traj[:, 0], traj[:, 1])

        # Draw collision zone for each drone
        zone = drones[i].get_collision_zone()
        
        # Handle circular collision zone
        if len(zone) == 2:
            center, radius = zone
            # Create and add circular patch to plot
            circ = plt.Circle(center, radius, color=colors[i], alpha=0.2)
            ax.add_patch(circ)
            
        # Handle rectangular collision zone
        elif len(zone) == 3:
            p1, p2, w = zone
            # Calculate rectangle parameters from two points and width
            dx, dy = np.array(p2) - np.array(p1)
            length = np.linalg.norm([dx, dy])
            angle = np.arctan2(dy, dx) * 180 / np.pi
            # Create and add rectangular patch to plot
            rect = plt.Rectangle(p1, length, w, angle=angle, color=colors[i], alpha=0.2)
            ax.add_patch(rect)

ani = animation.FuncAnimation(fig, update, frames=len(positions[0]), interval=100)
ani.save("simulace.gif", writer="pillow")
print("✅ Animace uložena jako simulace.gif")
