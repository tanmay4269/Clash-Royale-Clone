from game.utils import *
from game.entities.buildings.crown_tower import CrownTower
from game.entities.projectiles.arrow import Arrow
from game.entities.projectile_shooter import ProjectileShooter


@EntityRegistry.register("PrincessTower")
class PrincessTower(CrownTower, ProjectileShooter):
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
            hitpoints=2534,
            damage=90,
            attack_radius=7.5,
            hit_speed=0.8, first_hit_speed=0.8,
            **kwargs
        )

    
    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        self.render_projectiles(screen)


    def update(self, dt, arena_cell_occupancy) -> bool:
        if self.health < 0:
            self.owner.king_tower.is_activated = True
            return False
        
        self.update_projectiles(dt)

        return True


    def get_projectile(self, direction, target_types):
        return Arrow(self.owner, self.position.copy(), direction, self.damage, self.attack_radius_cells / 16, target_types)
