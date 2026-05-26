from game.utils import *
from game.entity import Entity
from game.entities.troop import Troop
from game.entities.projectile_shooter import ProjectileShooter
from game.entities.projectiles.arrow import Arrow


archer_entity_properties = {
    "deploy_cost": 3, 
    "deploy_delay": 1.0,
    "entity_type": EntityType.GROUND,
    "hitpoints": 304,
    "damage": 112,
    "attack_radius": 5,
    "hit_speed": 0.9, 
    "first_hit_speed": 0.5,
    "target_types": EntityType.get_all(),
}

archer_troop_shooter_properties = {
    "radius": 0.35,
    "speed": Troop.Speed.MEDIUM,
    "mass": 1.0,
}

class Archer(Troop, ProjectileShooter):
    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner, row, col,
            **archer_entity_properties,
            **archer_troop_shooter_properties,
        )

    def attack_mechanics(self) -> None:
        pass

    def render(self, screen) -> None:
        color = "red" if self.owner.side_index == 1 else LIGHT_BLUE
        super().render(screen, color)
        self.render_projectiles(screen)

        font = pygame.font.SysFont(None, 12)
        text = font.render("A", True, (0, 0, 0))
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
        return Arrow(
            self.owner,
            self.position.copy(),
            direction,
            self.damage,
            max_range=self.attack_radius_cells / 16,
            target_types=target_types,
        )


@EntityRegistry.register("Archers")
class Archers(Entity):
    _image_path = "assets/archers.png"

    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner, row, col,
            **archer_entity_properties,
        )
        self.position += Vector2(8, 8)
        self.units = []
        self.unit_radius = archer_troop_shooter_properties["radius"]
        
        if owner is not None:
            self.units = [
                Archer(owner, row, col - self.unit_radius - 0.1, **kwargs),
                Archer(owner, row, col + self.unit_radius, **kwargs)
            ]

    def get_units(self):
        return self.units

    def get_cell_occupancy_index(self):
        return 3

    def get_cell_occupancy(self):
        size = self.unit_radius * 16
        pos = self.position - Vector2(size, size)
        return np.ones([1, 1]) * self.get_cell_occupancy_index(), pos
