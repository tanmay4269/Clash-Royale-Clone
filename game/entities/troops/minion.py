from game.utils import *
from game.entity import Entity
from game.entities.troop import Troop
from game.entities.projectile_shooter import ProjectileShooter
from game.entities.projectiles.minion_spit import MinionSpit


minion_entity_properties = {
    "deploy_cost": 3,
    "deploy_delay": 1.0,
    "entity_type": EntityType.AIR,
    "hitpoints": 230,
    "damage": 107,
    "attack_radius": 2.5,
    "hit_speed": 1.2,
    "first_hit_speed": 0.5,
    "target_types": EntityType.get_all(),
}

minion_troop_shooter_properties = {
    "radius": 0.3,
    "speed": Troop.Speed.FAST,
    "mass": 0.8,
}


class Minion(Troop, ProjectileShooter):
    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner, row, col,
            **minion_entity_properties,
            **minion_troop_shooter_properties,
        )

    def attack_mechanics(self) -> None:
        pass

    def render(self, screen) -> None:
        color = "#5C3A86" if self.owner.side_index == 1 else "#3F51B5"
        super().render(screen, color)
        self.render_projectiles(screen)

        font = pygame.font.SysFont(None, 12)
        text = font.render("O", True, (255, 255, 255))
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
        return MinionSpit(
            self.owner,
            self.position.copy(),
            direction,
            self.damage,
            max_range=self.attack_radius_cells / 16,
            target_types=target_types,
        )


@EntityRegistry.register("Minions")
class Minions(Entity):
    _image_path = "assets/minions.png"

    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner, row, col,
            **minion_entity_properties,
        )
        self.units = []
        self.unit_radius = minion_troop_shooter_properties["radius"]

        if owner is not None:
            front_dir = 1.0 if owner.side_index == 1 else -1.0
            self.units = [
                Minion(owner, row + front_dir * 0.35, col, **kwargs),
                Minion(owner, row - front_dir * 0.25, col - 0.35, **kwargs),
                Minion(owner, row - front_dir * 0.25, col + 0.35, **kwargs),
            ]

    def get_units(self):
        return self.units

    def get_cell_occupancy_index(self):
        return 3

    def get_cell_occupancy(self):
        size = self.unit_radius * 16
        pos = self.position - Vector2(size, size)
        return np.ones([1, 1]) * self.get_cell_occupancy_index(), pos
