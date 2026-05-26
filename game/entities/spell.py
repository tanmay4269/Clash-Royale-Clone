from game.utils import *
from game.entity import Entity

class Spell(Entity):
    def __init__(
        self, owner, row: int, col: int,
        deploy_cost: int,
        deploy_delay: float,
        radius: float,
        damage: int,
        crown_tower_damage: int,
        **kwargs
    ):
        kwargs.pop('width', None)
        kwargs.pop('height', None)
        super().__init__(
            owner=owner, row=row, col=col,
            deploy_cost=deploy_cost, deploy_delay=deploy_delay,
            entity_type=EntityType.GROUND,
            hitpoints=1,
            damage=damage,
            attack_radius=radius,
            hit_speed=0.0, first_hit_speed=0.0,
            target_types=set(),
            **kwargs
        )
        self.radius = radius
        self.radius_cells = radius * 16
        self.crown_tower_damage = crown_tower_damage
        self.size = 0.0

        self.position = Vector2((col - 0.5) * 16, (row - 0.5) * 16)
        
        if owner is not None:
            self.start_position = owner.king_tower.position.copy()
        else:
            self.start_position = Vector2(0, 0)
        
        self.impact_done = False
        self.is_targetable = False
        self.explosion_timer = 0.0
        self.explosion_duration = 0.3

    def update(self, dt, arena_cell_occupancy) -> bool:
        if self.owner is None:
            return False

        if not self.impact_done:
            self.apply_spell_effect()
            self.impact_done = True
            self.explosion_timer = 0.0

        self.explosion_timer += dt
        return self.explosion_timer < self.explosion_duration

    def apply_spell_effect(self) -> None:
        if self.owner is None or self.owner.opponent is None:
            return

        from game.entities.buildings.crown_tower import CrownTower

        for obj in self.owner.opponent.objects:
            if not obj.has_deployed():
                continue

            delta = obj.position - self.position
            if isinstance(obj.size, Vector2):
                half_size = obj.size / 2
                closest_x = np.clip(delta.x, -half_size.x, half_size.x)
                closest_y = np.clip(delta.y, -half_size.y, half_size.y)
                closest_point = Vector2(closest_x, closest_y)
                dist = (delta - closest_point).length()
                if dist > self.radius_cells:
                    continue
            else:
                if delta.length() > self.radius_cells + obj.size:
                    continue

            damage_to_apply = self.crown_tower_damage if isinstance(obj, CrownTower) else self.damage
            obj.apply_damage(damage_to_apply)

    def get_cell_occupancy_index(self) -> int:
        return 0

    def get_cell_occupancy(self):
        return np.zeros((1, 1)), Vector2(0, 0)
