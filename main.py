import random



class Container:
    def __init__(self, length, width, height):
        self.l = length
        self.w = width
        self.h = height
        self.x = None
        self.y = None
        self.z = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.inserted = None

    def place_randomly(self, Wx, Wy, Wz) -> None:
        self.x = random.randint(0, Wx-1)
        self.y = random.randint(0, Wy-1)
        self.z = random.randint(0, Wz-1)
        dimensions = [self.l, self.w, self.h]
        random.shuffle(dimensions)
        self.dx, self.dy, self.dz = dimensions 
    
    def fits_in_magazine(self, Wx:int, Wy:int, Wz:int) -> bool:
        fits_on_x = (self.x) >= 0 and ((self.x + self.dx) <= Wx)
        fits_on_y = (self.y) >= 0 and ((self.y + self.dy) <= Wy)
        fits_on_z = (self.z) >= 0 and ((self.z + self.dz) <= Wz)
        return (fits_on_x and fits_on_y and fits_on_z) # musi miescic sie w kazdym wymiarze
    
    def doesnt_overlap(self, other:"Container") -> bool:
        # w tych 3 linijkach ponizej zaprzeczamy warunek mowiacy o nienakladaniu sie na poszczegolnych osiach
        overlaps_on_x = not ((self.x + self.dx <= other.x) or (other.x + other.dx <= self.x))
        overlaps_on_y = not ((self.y + self.dy <= other.y) or (other.y + other.dy <= self.y))
        overlaps_on_z = not ((self.z + self.dz <= other.z) or (other.z + other.dz <= self.z))
        #zaprzeczamy warunek mowiacy o nakladaniu sie w przestrzeni
        return not (overlaps_on_x and overlaps_on_y and overlaps_on_z)
    
    def mutation(self, prob_of_moving: float, prob_of_rotation: float, prob_of_presence: float):
        pass
        # if (random.random() < prob_of_moving):
