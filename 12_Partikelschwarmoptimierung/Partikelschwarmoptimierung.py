import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==================================================
# Ackley-Funktion
# ==================================================

def objective_function(position):
    x = position[..., 0]
    y = position[..., 1]

    a = 20
    b = 0.2
    c = 2 * np.pi

    return (
        -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
        - np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
        + a
        + np.e
    )

# ==================================================
# Parameter
# ==================================================

np.random.seed(42)

num_particles = 30
num_dimensions = 2
max_iterations = 100

w = 0.7
c1 = 1.5
c2 = 1.5

bounds = [-2, 2]

# ==================================================
# Hintergrund vorbereiten
# ==================================================

x_grid = np.linspace(bounds[0], bounds[1], 500)
y_grid = np.linspace(bounds[0], bounds[1], 500)

X, Y = np.meshgrid(x_grid, y_grid)

grid = np.stack((X, Y), axis=-1)
Z = objective_function(grid)

# ==================================================
# Initialisierung
# ==================================================

positions = np.random.uniform(
    bounds[0],
    bounds[1],
    (num_particles, num_dimensions)
)

velocities = np.random.uniform(
    -0.2,
    0.2,
    (num_particles, num_dimensions)
)

personal_best_positions = positions.copy()
personal_best_scores = objective_function(personal_best_positions)

best_idx = np.argmin(personal_best_scores)

global_best_position = personal_best_positions[best_idx].copy()
global_best_score = personal_best_scores[best_idx]

trajectory = [positions.copy()]
global_best_history = [global_best_position.copy()]
best_values = [global_best_score]

# ==================================================
# PSO
# ==================================================

for iteration in range(max_iterations):

    fitness = objective_function(positions)

    better_mask = fitness < personal_best_scores

    personal_best_positions[better_mask] = positions[better_mask]
    personal_best_scores[better_mask] = fitness[better_mask]

    best_idx = np.argmin(personal_best_scores)

    if personal_best_scores[best_idx] < global_best_score:
        global_best_position = personal_best_positions[best_idx].copy()
        global_best_score = personal_best_scores[best_idx]

    r1 = np.random.rand(num_particles, num_dimensions)
    r2 = np.random.rand(num_particles, num_dimensions)

    velocities = (
        w * velocities
        + c1 * r1 * (personal_best_positions - positions)
        + c2 * r2 * (global_best_position - positions)
    )

    positions = positions + velocities

    positions = np.clip(
        positions,
        bounds[0],
        bounds[1]
    )   

    trajectory.append(positions.copy())
    global_best_history.append(global_best_position.copy())
    best_values.append(global_best_score)

    # SVG Export alle 5 Iterationen
    if (iteration + 1) % 5 == 0:

        fig_export, ax_export = plt.subplots(figsize=(8, 7))

        contour = ax_export.contourf(
            X,
            Y,
            Z,
            levels=100,
            cmap="viridis"
        )

        plt.colorbar(contour)

        ax_export.scatter(
            0,
            0,
            color="white",
            marker="*",
            s=250,
            label="Theoretisches Minimum"
        )

        ax_export.scatter(
            positions[:, 0],
            positions[:, 1],
            color="red",
            s=40,
            label="Partikel"
        )

        ax_export.scatter(
            global_best_position[0],
            global_best_position[1],
            color="cyan",
            marker="*",
            s=250,
            label="Bestes gefundenes Minimum"
        )

        ax_export.set_title(
            f"Ackley-Funktion - Iteration {iteration + 1}"
        )

        ax_export.set_xlabel("x")
        ax_export.set_ylabel("y")

        ax_export.legend()

        plt.tight_layout()

        plt.savefig(
            f"ackley_iteration_{iteration + 1:03d}.svg",
            format="svg",
            bbox_inches="tight"
        )

        plt.close(fig_export)

# ==================================================
# Ergebnis
# ==================================================

x_best = global_best_position[0]
y_best = global_best_position[1]

print("\nGefundenes Minimum:")
print(f"x = {x_best:.8f}")
print(f"y = {y_best:.8f}")
print(f"f(x,y) = {global_best_score:.12f}")

# ==================================================
# Hauptfenster
# ==================================================

fig, ax = plt.subplots(figsize=(9, 7))

contour = ax.contourf(
    X,
    Y,
    Z,
    levels=100,
    cmap="viridis"
)

plt.colorbar(
    contour,
    label="Funktionswert"
)

ax.scatter(
    0,
    0,
    color="white",
    marker="*",
    s=250,
    label="Theoretisches Minimum"
)

best_marker = ax.scatter(
    [global_best_history[0][0]],
    [global_best_history[0][1]],
    color="cyan",
    marker="*",
    s=250,
    label="Bestes gefundenes Minimum"
)

scat = ax.scatter(
    trajectory[0][:, 0],
    trajectory[0][:, 1],
    color="red",
    s=40,
    label="Partikel"
)

ax.legend()

# ==================================================
# Animation
# ==================================================

def update(frame):

    scat.set_offsets(trajectory[frame])

    best_marker.set_offsets([
        [
            global_best_history[frame][0],
            global_best_history[frame][1]
        ]
    ])

    ax.set_title(
        f"Ackley-Funktion mit PSO\n"
        f"Iteration: {frame}/{max_iterations}\n"
        f"Bestes f(x,y): {best_values[frame]:.6f}"
    )

    return scat, best_marker

ani = FuncAnimation(
    fig,
    update,
    frames=len(trajectory),
    interval=150,
    blit=False
)

plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()