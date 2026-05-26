from game.utils import *
from game.entities.projectile import Projectile


class Arrow(Projectile):
    RADIUS = 0.25
    SPEED  = 200

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
            self.RADIUS,

            start_position, 
            direction, 
            speed=self.SPEED, 

            projectile_type=Projectile.TargetType.SINGLE,
            damage=damage,
            max_range=max_range,
            target_types=target_types
        )
    
    def render(self, screen) -> None:
        if self.velocity.length_squared() > 0:
            direction = self.velocity.normalize()
        else:
            direction = Vector2(0, -1)

        length = 12
        width = 3
        right = Vector2(-direction.y, direction.x)

        p1 = self.position - direction * (length / 2) - right * (width / 2)
        p2 = self.position + direction * (length / 2) - right * (width / 2)
        p3 = self.position + direction * (length / 2) + right * (width / 2)
        p4 = self.position - direction * (length / 2) + right * (width / 2)

        pygame.draw.polygon(screen, "black", [p1, p2, p3, p4])
