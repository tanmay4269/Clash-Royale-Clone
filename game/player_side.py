from game.entities.buildings.king_tower import KingTower
from game.entities.buildings.princess_tower import PrincessTower
from typing import List


class PlayerSide:
    def __init__(self):
        self.side_index = None
        self.opponent = None

        self.king_tower = None
        self.princess_tower_1 = None  # The one closer to (0, 0) 
        self.princess_tower_2 = None

        self.elixirs = 5.0  # init
        self.max_elixirs = 10  # TODO: move this up to arena class, pass as arg
        self.elixirs_increment_cooldown = 3.0
        self._elixir_timer = 0.0

        self.objects: List = []  # Add towers once initialised
            # TODO: make a dict instead, but need to change getters and setters


    def update(self, dt):
        self._elixir_timer += dt

        if self._elixir_timer < self.elixirs_increment_cooldown:
            return

        self._elixir_timer = 0.0
        self.elixirs = min(self.max_elixirs, self.elixirs + 1)

    
    def spend_elixirs(self, amount):
        self.elixirs -= amount


    def set_double_elixir_mode(self):
        self.elixirs_increment_cooldown = 1.5

    
    def set_tripple_elixir_mode(self):
        self.elixirs_increment_cooldown = 1.0
    

    def get_objects(self):
        return self.objects


    def add_object(self, obj):
        if obj not in self.objects:
            self.objects.append(obj)


    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)
            del obj


    def set_opponent(self, opponent):
        assert opponent is not self
        self.opponent = opponent


# The one closer to (0, 0)
class PlayerSide1(PlayerSide):
    def __init__(self):
        super().__init__()
        self.side_index = 1

        self.king_tower = KingTower(self, 3, 18/2, 2.5, 2.5)
        self.princess_tower_1 = PrincessTower(self, 6.5, 3.5, 2, 2)
        self.princess_tower_2 = PrincessTower(self, 6.5, 18 - 3.5, 2, 2)

        # Deterministic order: king first, then princess_1, princess_2
        self.objects = [self.king_tower, self.princess_tower_1, self.princess_tower_2]


class PlayerSide2(PlayerSide):
    def __init__(self):
        super().__init__()
        self.side_index = 2

        self.king_tower = KingTower(self, 32 - 3, 18/2, 2.5, 2.5)
        self.princess_tower_1 = PrincessTower(self, 32 - 6.5, 3.5, 2, 2)
        self.princess_tower_2 = PrincessTower(self, 32 - 6.5, 18 - 3.5, 2, 2)
        
        # Deterministic order: king first, then princess_1, princess_2
        self.objects = [self.king_tower, self.princess_tower_1, self.princess_tower_2]
