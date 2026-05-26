from game.utils import *
from game.entities.projectile import Projectile


class Arrow(Projectile):
    def __init__(
        self, 
        owner, 

        start_position: Vector2, 
        direction: Vector2, 

        damage:float,
        max_range: float,
        target_types: Set[EntityType],
    ):
        super().__init__(
            owner, 
            0.25,  # Radius

            start_position, 
            direction, 
            speed=200, 

            projectile_type=Projectile.TargetType.SINGLE,
            damage=damage,
            max_range=max_range,
            target_types=target_types
        )
    
    def render(self, screen) -> None:
        pygame.draw.circle(screen, "black", self.position, 4)
