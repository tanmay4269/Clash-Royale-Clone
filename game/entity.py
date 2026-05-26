from typing import Tuple, List, Set
from rich import print

import numpy as np

from pygame import Vector2

# from game.utils import *


class EntityType:
    GROUND      = 0
    AIR         = 1
    BUILDING    = 2

    def get_all():
        return set({
            EntityType.GROUND,
            EntityType.AIR,
            EntityType.BUILDING,
        })

    def num_types():
        return 3


class EntityRegistry:
    _registry = {}
    _dummy_instances = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            cls._dummy_instances[name] = subclass(None, -1, -1, width=-1, height=-1)
            return subclass
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    @classmethod
    def all(cls):
        return cls._registry.values()

    @classmethod
    def aggregate(cls, attr):
        values = [getattr(inst, attr) for inst in cls._dummy_instances.values() if hasattr(inst, attr)]
        return {
            "mean": sum(values) / len(values) if values else 0,
            "min": min(values),
            "max": max(values),
        }


class Entity:
    # TODO: make a child class called "Card" that defines abstraction for buildings and troops 
    #   while projectile can be merged as a child of this class to stay DRY

    CELL_OCCUPANCY_LAYERS = 3  # Excluding the trivial 0th layer

    @classmethod
    def get_image_path(cls) -> str | None:
        return getattr(cls, "_image_path", None)

    def __init__(
        self, owner, row, col,
        deploy_cost, deploy_delay,
        entity_type: EntityType,
        hitpoints,
        damage,
        attack_radius,
        hit_speed, first_hit_speed,
        target_types: Set[EntityType],
        **kwargs
    ):
        """
        owner: of type PlayerSide
        row, col: in tiles
        deploy_cost: in elixirs
        deploy_delay: in sec. Time between placing an entity and it actually starting 
            to do its thing. Nothing interacts with it in this phase.
        entity_type: of type EntityType
        hitpoints: health
        damage: attack damage per shot
        hit_speed: in sec
        first_hit_speed: in sec
        target_types: set of EntityType, only these types
            of enemies will be targeted
        """
        super().__init__(**kwargs)

        # Abstract Attributes
        self.owner = owner
        self.is_alive = True
        self.is_targetable = True

        # Physical Attributes

        # (row, col) is the entity's centre
        self.row = row
        self.col = col

        self.position = Vector2(col * 16, row * 16)  # TODO: remove the hardcoding

        self.entity_type = entity_type
        self.deploy_cost = deploy_cost
        self.deploy_delay = deploy_delay

        # Attack Attributes
        self.hitpoints = hitpoints  # The maximum health
        self.health = hitpoints     # The real time health

        self.damage = damage
        self.attack_radius_cells = attack_radius * 16

        self.visibility_cells = 18//3 * 16  # Radius within which this entity can potentially agro on an opponent, else defaults to opponent's towers
        
        self.hit_speed = hit_speed
        self.first_hit_speed = first_hit_speed
        
        self.target_types = target_types  # Set of entity types
        if not isinstance(self.target_types, set):
            self.target_types = set({self.target_types})

        # Private trackers
        self._deploy_timer = 0
        self._attack_timer = 0
        self._is_first_hit = False


    def render(self, screen) -> None:
        raise NotImplementedError


    def has_deployed(self, dt=0.0) -> bool:
        """
        To be called before self.update has ever been called
        
        dt: used to update the timer
        """
        if self._deploy_timer < self.deploy_delay:
            self._deploy_timer += dt
            return False
        else:
            return True


    def update(self, dt, arena_cell_occupancy) -> bool:
        """
        return: True means alive
        """
        raise NotImplementedError


    def get_cell_occupancy_index(self) -> int:
        """
        0 => Unoccupied
        1 => Permanent occupancy
        2 => Building
        3 => Troop
        """
        return 0


    def get_cell_occupancy(self):
        """
        occupancy grid and top left position
        """
        raise NotImplementedError


    ########################
    ### Combat Mechanics ###
    ########################

    def attack_mechanics(self) -> None:
        raise NotImplementedError


    def apply_damage(self, damage) -> None:
        """
        blindly trust whoever called this!
        """
        self.health -= damage
