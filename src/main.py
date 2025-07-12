import matplotlib.pyplot as plt
import numpy as np
from drone_factory import create_drone

drones = [
    create_drone("quadcopter", [0., 0.], 0.),
    create_drone("fixedwing", [30., 0.], 90.)
]

for d in drones:
    d.dt = 0.5
    d.flight_time = 0.0

actions = [
    [0.5, 0.0],
    0
]

positions = {i: [] for i in range(len(drones))}

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

# Simulace
for step in range(60):
    print(f"\nSTEP {step}")
    for i, di in enumerate(drones):
        for j, dj in enumerate(drones):
            if i != j and check_collision(di, dj):
                print(f"⚠️  KOLIZE mezi {i} a {j}!")

    for i, drone in enumerate(drones):
        drone.move(actions[i])
        positions[i].append(drone.position.copy())

# Vizualizace
fig, ax = plt.subplots()
colors = ["blue", "red"]
labels = ["Quad", "FixedWing"]

for i, pos in positions.items():
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    ax.plot(xs, ys, label=labels[i], color=colors[i])

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

ax.legend()
ax.grid(True)
ax.set_aspect('equal')
plt.title("Drony s kolizními zónami")
plt.savefig("trajektorie.png")
