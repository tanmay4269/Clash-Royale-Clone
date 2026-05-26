from game.utils import *
from game.entities.buildings.crown_tower import CrownTower
from game.entities.projectiles.cannon_shell import CannonShell
from game.entities.projectile_shooter import ProjectileShooter


@EntityRegistry.register("KingTower")
class KingTower(CrownTower, ProjectileShooter):
    def __init__(
        self, owner, row: int, col: int, 
        width: int, height: int,
        **kwargs
    ):
        """
        Level 9
        """
        super().__init__(
            owner=owner, row=row, col=col,
            width=width, height=height,
            hitpoints=4008,
            damage=90,
            attack_radius=7,
            hit_speed=1.0, first_hit_speed=0.0,
            **kwargs
        )

        self.is_activated = False

    
    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        self.render_projectiles(screen)


    def update(self, dt, arena_cell_occupancy) -> bool:
        if self.health < 0:
            return False

        if self.is_activated:
            self.update_projectiles(dt)

        return True


    def get_projectile(self, direction, target_types):
        return CannonShell(
            self.owner, 
            self.position.copy(), 
            direction, 
            self.damage, 
            self.attack_radius_cells / 16, 
            target_types
        )
