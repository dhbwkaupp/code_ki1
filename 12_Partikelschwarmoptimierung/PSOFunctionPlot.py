import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Ackley-Funktion
# ==========================================

def ackley(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi

    return (
        -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
        - np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
        + a
        + np.e
    )

# ==========================================
# Gitter
# ==========================================

x = np.linspace(-2, 2, 300)
y = np.linspace(-2, 2, 300)

X, Y = np.meshgrid(x, y)
Z = ackley(X, Y)

# ==========================================
# Perspektiven
# ==========================================

views = [
    {
        "filename": "ackley_isometrisch.svg",
        "title": "Ackley-Funktion (isometrisch)",
        "elev": 30,
        "azim": -135,
    },
    {
        "filename": "ackley_schraeg_oben.svg",
        "title": "Ackley-Funktion (schräg von oben)",
        "elev": 65,
        "azim": -90,
    },
    {
        "filename": "ackley_draufsicht.svg",
        "title": "Ackley-Funktion (Draufsicht)",
        "elev": 89,
        "azim": -90,
    },
]

# ==========================================
# Plotten & Exportieren
# ==========================================

for view in views:

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )


    ax.set_title(view["title"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    ax.view_init(
        elev=view["elev"],
        azim=view["azim"]
    )

    fig.colorbar(
        surface,
        shrink=0.7,
        aspect=18,
        label="Funktionswert"
    )

    ax.legend()

    plt.tight_layout()

    # SVG speichern
    plt.savefig(
        view["filename"],
        format="svg",
        bbox_inches="tight"
    )

    # Plot anzeigen
    plt.show()

    # Fenster schließen
    plt.close(fig)

    print(f"Exportiert: {view['filename']}")

print("Alle SVG-Dateien wurden erzeugt.")