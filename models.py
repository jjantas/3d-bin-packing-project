from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

Dims = Tuple[int, int, int]


@dataclass
class Container:
    l: int
    w: int
    h: int

    x: Optional[int] = None
    y: Optional[int] = None
    z: Optional[int] = None

    dx: Optional[int] = None
    dy: Optional[int] = None
    dz: Optional[int] = None

    inserted: Optional[bool] = None

    Wx: Optional[int] = None
    Wy: Optional[int] = None
    Wz: Optional[int] = None

    def copy(self) -> "Container":
        c = Container(self.l, self.w, self.h)
        c.x, c.y, c.z = self.x, self.y, self.z
        c.dx, c.dy, c.dz = self.dx, self.dy, self.dz
        c.inserted = self.inserted
        c.Wx, c.Wy, c.Wz = self.Wx, self.Wy, self.Wz
        return c

    def pick_warehouse(self, Wx: int, Wy: int, Wz: int) -> None:
        self.Wx, self.Wy, self.Wz = Wx, Wy, Wz

    def choose_presence_randomly(self, prob_of_presence: float) -> None:
        self.inserted = random.random() < prob_of_presence

    def choose_rotation_randomly(self) -> None:
        dimensions = [self.l, self.w, self.h]
        random.shuffle(dimensions)
        self.dx, self.dy, self.dz = dimensions

    def place_randomly(self, bias_inside: bool = True) -> None:
        if self.Wx is None or self.Wy is None or self.Wz is None:
            raise ValueError("You need to pick a warehouse first!")
        if self.dx is None or self.dy is None or self.dz is None:
            raise ValueError("You need to choose rotation first!")

        def rand_pos(W: int, d: int) -> int:
            if not bias_inside:
                return random.randint(0, max(0, W - 1))
            if W >= d:
                return random.randint(0, W - d)
            return random.randint(0, max(0, W - 1))

        self.x = rand_pos(self.Wx, self.dx)
        self.y = rand_pos(self.Wy, self.dy)
        self.z = rand_pos(self.Wz, self.dz)

    def fits_in_magazine(self) -> bool:
        if self.Wx is None or self.Wy is None or self.Wz is None:
            raise ValueError("You need to pick a warehouse first!")
        if None in (self.x, self.y, self.z, self.dx, self.dy, self.dz):
            raise ValueError("Container not fully initialized (position/rotation).")

        fits_on_x = self.x >= 0 and (self.x + self.dx) <= self.Wx
        fits_on_y = self.y >= 0 and (self.y + self.dy) <= self.Wy
        fits_on_z = self.z >= 0 and (self.z + self.dz) <= self.Wz
        return fits_on_x and fits_on_y and fits_on_z

    def overlaps_xy(self, other: "Container") -> bool:
        overlaps_on_x = not (
            (self.x + self.dx <= other.x) or (other.x + other.dx <= self.x)
        )
        overlaps_on_y = not (
            (self.y + self.dy <= other.y) or (other.y + other.dy <= self.y)
        )
        return overlaps_on_x and overlaps_on_y

    def doesnt_overlap(self, other: "Container") -> bool:
        if None in (self.x, self.y, self.z, self.dx, self.dy, self.dz):
            raise ValueError("Self container not fully initialized.")
        if None in (other.x, other.y, other.z, other.dx, other.dy, other.dz):
            raise ValueError("Other container not fully initialized.")

        overlaps_on_x = not (
            (self.x + self.dx <= other.x) or (other.x + other.dx <= self.x)
        )
        overlaps_on_y = not (
            (self.y + self.dy <= other.y) or (other.y + other.dy <= self.y)
        )
        overlaps_on_z = not (
            (self.z + self.dz <= other.z) or (other.z + other.dz <= self.z)
        )
        return not (overlaps_on_x and overlaps_on_y and overlaps_on_z)

    def move_randomly(self, strength: float) -> None:
        if self.Wx is None or self.Wy is None or self.Wz is None:
            raise ValueError("You need to pick a warehouse first!")
        if None in (self.x, self.y, self.z):
            raise ValueError("You need to place the container first!")

        max_x = max(0, self.Wx - 1)
        max_y = max(0, self.Wy - 1)
        max_z = max(0, self.Wz - 1)

        x_shift = round((((random.random() * 2) - 1) * strength * self.Wx))
        y_shift = round((((random.random() * 2) - 1) * strength * self.Wy))
        z_shift = round((((random.random() * 2) - 1) * strength * self.Wz))

        self.x = min(max(self.x + x_shift, 0), max_x)
        self.y = min(max(self.y + y_shift, 0), max_y)
        self.z = min(max(self.z + z_shift, 0), max_z)

    def mutation(
        self,
        prob_of_moving: float,
        prob_of_rotation: float,
        prob_of_presence: float,
        mutation_strength: float,
    ) -> None:
        if random.random() < prob_of_moving:
            self.move_randomly(mutation_strength)
        if random.random() < prob_of_rotation:
            self.choose_rotation_randomly()
        if random.random() < prob_of_presence:
            if self.inserted is None:
                self.inserted = True
            else:
                self.inserted = not self.inserted


@dataclass
class Solution:
    containers: List[Container]

    def copy(self) -> "Solution":
        return Solution([c.copy() for c in self.containers])

    def is_feasible(self) -> bool:
        for c in self.containers:
            if c.inserted and (not c.fits_in_magazine()):
                return False

        for i in range(len(self.containers)):
            c1 = self.containers[i]
            if not c1.inserted:
                continue
            for j in range(i + 1, len(self.containers)):
                c2 = self.containers[j]
                if c2.inserted and (not c1.doesnt_overlap(c2)):
                    return False

        inserted = [c for c in self.containers if c.inserted]
        for c in inserted:
            if c.z == 0:
                continue
            supported = False
            for b in inserted:
                if b is c:
                    continue
                if (b.z + b.dz) == c.z and c.overlaps_xy(b):
                    supported = True
                    break
            if not supported:
                return False

        return True
