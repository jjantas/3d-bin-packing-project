# src/binpack3d/viz.py
from __future__ import annotations
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from models import Solution, Dims


def _cuboid_faces(x, y, z, dx, dy, dz):
    # 8 vertices
    v = [
        (x, y, z),
        (x + dx, y, z),
        (x + dx, y + dy, z),
        (x, y + dy, z),
        (x, y, z + dz),
        (x + dx, y, z + dz),
        (x + dx, y + dy, z + dz),
        (x, y + dy, z + dz),
    ]
    # 6 faces
    return [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[3], v[0], v[4], v[7]],
    ]


def plot_solution(sol: Solution, warehouse: Dims, title: str = "") -> None:
    Wx, Wy, Wz = warehouse
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, Wx)
    ax.set_ylim(0, Wy)
    ax.set_zlim(0, Wz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)

    # draw boxes
    for c in sol.containers:
        if not c.inserted:
            continue
        faces = _cuboid_faces(c.x, c.y, c.z, c.dx, c.dy, c.dz)
        poly = Poly3DCollection(faces, alpha=0.25)
        ax.add_collection3d(poly)

    plt.show()
