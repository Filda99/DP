import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from drone_factory import create_drone

# === Drony a cíle ===
drones = [
    create_drone("quadcopter", [0., 0.], 0.),
    create_drone("fixedwing", [50., 0.], 90.)
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
def quad_action(drone, goal, avoid=False):
    if avoid:
        return [-1.0, 1.0]
    vec = np.array(goal) - np.array(drone.position)
    norm = np.linalg.norm(vec)
    if norm < 1.0:
        return [0, 0]
    return (vec / norm).tolist()

def fixedwing_action(drone, goal, avoid=False):
    import math
    if avoid:
        return -15
    pos = np.array(drone.position)
    vec = np.array(goal) - pos
    target_angle = math.degrees(math.atan2(vec[1], vec[0]))
    delta = (target_angle - drone.heading + 360) % 360
    if delta > 180:
        delta -= 360
    delta = max(min(delta, 15), -15)
    return delta

# === Kolizní kontrola ===
def check_collision(di, dj):
    zi = di.get_collision_zone()
    zj = dj.position
    if len(zi) == 2:
        center, radius = zi
        return np.linalg.norm(np.array(center) - np.array(zj)) <= radius
    elif len(zi) == 3:
        p1, p2, width = zi
        a, b, p = np.array(p1), np.array(p2), np.array(zj)
        ab, ap = b - a, p - a
        proj = np.dot(ap, ab) / np.dot(ab, ab)
        closest = a + proj * ab
        dist = np.linalg.norm(closest - p)
        return 0 <= proj <= 1 and dist <= width / 2
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

def update(frame):
    ax.set_title(f"Krok {frame} {'⚠️ Kolize!' if frame in collisions else ''}")
    for patch in reversed(ax.patches):
        patch.remove()

    for i in range(len(drones)):
        traj = np.array(positions[i][:frame+1])
        if len(traj) > 1:
            scatters[i].set_data(traj[:, 0], traj[:, 1])

        # kolizní zóna
        zone = drones[i].get_collision_zone()
        if len(zone) == 2:
            center, radius = zone
            circ = plt.Circle(center, radius, color=colors[i], alpha=0.2)
            ax.add_patch(circ)
        elif len(zone) == 3:
            p1, p2, w = zone
            dx, dy = np.array(p2) - np.array(p1)
            length = np.linalg.norm([dx, dy])
            angle = np.arctan2(dy, dx) * 180 / np.pi
            rect = plt.Rectangle(p1, length, w, angle=angle, color=colors[i], alpha=0.2)
            ax.add_patch(rect)

ani = animation.FuncAnimation(fig, update, frames=len(positions[0]), interval=100)
ani.save("simulace.gif", writer="pillow")
print("✅ Animace uložena jako simulace.gif")
