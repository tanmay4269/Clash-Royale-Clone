from game.utils import *
from game.entities.projectile import Projectile


class MinionSpit(Projectile):
    RADIUS = 0.2
    SPEED  = 200

    def __init__(
        self,
        owner,
        start_position: Vector2,
        direction: Vector2,
        damage: float,
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
            target_types=target_types,
        )

    def render(self, screen) -> None:
        pygame.draw.circle(screen, "#8A2BE2", self.position, 3)
        pygame.draw.circle(screen, "black", self.position, 3, width=1)
