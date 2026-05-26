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

        # Deck and hand initialization
        self.deck = []
        self.hand = []
        self.next_card = None
        self.active_card_idx = None
        self.init_deck()

    def init_deck(self):
        from game.entities.troops.knight     import Knight
        from game.entities.troops.giant      import Giant
        from game.entities.troops.mini_pekka import MiniPEKKA
        from game.entities.troops.musketeer  import Musketeer
        from game.entities.troops.archer     import Archers
        import random

        self.deck = [Knight, Giant, MiniPEKKA, Musketeer, Archers]
        shuffled = list(self.deck)
        random.shuffle(shuffled)
        self.hand = shuffled[:4]
        self.next_card = shuffled[4]
        self.active_card_idx = None

    def use_card(self, card_idx):
        """Replaces the used card in the hand with the next card, and draws a new next card."""
        if card_idx is None or not (0 <= card_idx < len(self.hand)):
            return
        
        used_card = self.hand[card_idx]
        
        # Move next_card into hand at card_idx
        self.hand[card_idx] = self.next_card
        
        # Sample a new next_card from deck that is NOT in the hand.
        import random
        pool = [c for c in self.deck if c not in self.hand]
        if pool:
            self.next_card = random.choice(pool)
        else:
            self.next_card = used_card
        
        # Clear active card selection
        self.active_card_idx = None


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
        if hasattr(obj, "get_units"):
            for unit in obj.get_units():
                if unit not in self.objects:
                    self.objects.append(unit)
        else:
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
