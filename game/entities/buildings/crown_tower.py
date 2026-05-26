from game.utils import *
from game.entities.building import Building


class CrownTower(Building):
    def __init__(
        self, owner, row: int, col: int, 
        width: int, height: int,
        hitpoints,
        damage,
        attack_radius,
        hit_speed, first_hit_speed,
        **kwargs
    ):
        super().__init__(
            owner=owner, row=row, col=col,
            deploy_cost=0, deploy_delay=0,
            entity_type=EntityType.BUILDING,
            width=width, height=height,
            hitpoints=hitpoints,
            damage=damage,
            attack_radius=attack_radius,
            hit_speed=hit_speed, first_hit_speed=first_hit_speed,
            target_types=EntityType.get_all(),
            **kwargs
        )

    
