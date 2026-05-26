from game.utils import *
from game.entities.troop import Troop
from game.entities.projectile_shooter import ProjectileShooter
from game.entities.projectiles.shotgun_bullet import ShotgunBullet


@EntityRegistry.register("Musketeer")
class Musketeer(Troop, ProjectileShooter):
    def __init__(self, owner, row, col, **kwargs):
        """Level 11"""
        super().__init__(
            owner, row, col,
            deploy_cost=4, deploy_delay=1.0,
            entity_type=EntityType.GROUND,
            radius=0.5,
            speed=Troop.Speed.MEDIUM,
            mass=1.0,
            hitpoints=721,
            damage=217,
            attack_radius=6,
            hit_speed=1.0, first_hit_speed=0.7,
            target_types=EntityType.get_all(),
        )

    
    def attack_mechanics(self) -> None:
        pass  # Intentionally suppressed; shooting is handled in update()


    def render(self, screen) -> None:
        color = "red" if self.owner.side_index == 1 else LIGHT_BLUE

        super().render(screen, color)
        self.render_projectiles(screen)

        font = pygame.font.SysFont(None, 12)
        text = font.render("M", True, (0, 0, 0))
        screen.blit(text, self.position - Vector2(3, 3))


    def update(self, dt, arena_cell_occupancy) -> bool:
        if self.health < 0:
            return False

        alive = Troop.update(self, dt, arena_cell_occupancy)
        if not alive:
            return False

        self.update_projectiles(dt)

        return True


    def get_projectile(self, direction, target_types):
        return ShotgunBullet(
            self.owner,
            self.position.copy(),
            direction,
            self.damage,
            max_range=self.attack_radius_cells / 16,
            target_types=target_types,
        )
