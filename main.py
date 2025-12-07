import random



class Container:
    def __init__(self, length, width, height) -> None:
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
        self.Wx = None
        self.Wy = None
        self.Wz = None

    def pick_warehouse(self, Wx:int, Wy:int, Wz:int):
        self.Wx = Wx
        self.Wy = Wy
        self.Wz = Wz

    def choose_presence_randomly(self, prob_of_presence:float):
        if random.random() < prob_of_presence:
            self.inserted = True
        else:
            self.inserted = False

    def choose_rotation_randomly(self):
        dimensions = [self.l, self.w, self.h]
        random.shuffle(dimensions)
        self.dx, self.dy, self.dz = dimensions

    def place_randomly(self) -> None:
        
        if (self.Wx is None) or (self.Wy is None) or (self.Wz is None):
            raise ValueError("You need to pick a container first!")

        self.x = random.randint(0, self.Wx-1)
        self.y = random.randint(0, self.Wy-1)
        self.z = random.randint(0, self.Wz-1)
    
    def fits_in_magazine(self) -> bool:
        
        if (self.Wx is None) or (self.Wy is None) or (self.Wz is None):
            raise ValueError("You need to pick a container first!")
        
        fits_on_x = (self.x) >= 0 and ((self.x + self.dx) <= self.Wx)
        fits_on_y = (self.y) >= 0 and ((self.y + self.dy) <= self.Wy)
        fits_on_z = (self.z) >= 0 and ((self.z + self.dz) <= self.Wz)
        return (fits_on_x and fits_on_y and fits_on_z) # musi miescic sie w kazdym wymiarze
    
    def doesnt_overlap(self, other:"Container") -> bool:
        # w tych 3 linijkach ponizej zaprzeczamy warunek mowiacy o nienakladaniu sie na poszczegolnych osiach
        overlaps_on_x = not ((self.x + self.dx <= other.x) or (other.x + other.dx <= self.x))
        overlaps_on_y = not ((self.y + self.dy <= other.y) or (other.y + other.dy <= self.y))
        overlaps_on_z = not ((self.z + self.dz <= other.z) or (other.z + other.dz <= self.z))
        #zaprzeczamy warunek mowiacy o nakladaniu sie w przestrzeni
        return not (overlaps_on_x and overlaps_on_y and overlaps_on_z)

    def move_randomly(self, strength:float) -> None:
        
        if (self.Wx is None) or (self.Wy is None) or (self.Wz is None):
            raise ValueError("You need to pick a container first!")

        while True:
            x_shift = round((((random.random()*2)-1) * strength * self.Wx))
            new_x_placement = self.x + x_shift
            if new_x_placement >= 0 and new_x_placement <= self.Wx:
                self.x = new_x_placement
                break
        while True:
            y_shift = round((((random.random()*2)-1) * strength * self.Wy))
            new_y_placement = self.y + y_shift
            if new_y_placement >= 0 and new_y_placement <= self.Wy:
                self.y = new_y_placement
                break
        while True:
            z_shift = round((((random.random()*2)-1) * strength * self.Wz))
            new_z_placement = self.z + z_shift
            if new_z_placement >= 0 and new_z_placement <= self.Wz:
                self.z = new_z_placement
                break
        

    def mutation(self, prob_of_moving: float, prob_of_rotation: float, prob_of_presence: float, mutation_strength: float):
        if (random.random() < prob_of_moving):
            self.move_randomly(mutation_strength)

        if (random.random() < prob_of_rotation):
            self.choose_rotation_randomly()

        if (random.random() < prob_of_presence):
            self.inserted = not self.inserted
        

