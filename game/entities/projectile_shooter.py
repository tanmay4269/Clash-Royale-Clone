from game.utils import *
from game.entity import Entity


class ProjectileShooter(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.projectiles: List = []


    def render_projectiles(self, screen):
        for proj in self.projectiles:
            proj.render(screen)


    def update_projectiles(self, dt):
        # Snapshot dead projectiles first, then remove: avoids mutating list while iterating
        dead_projectiles = [proj for proj in self.projectiles if not proj.update(dt)]
        for proj in dead_projectiles:
            self.projectiles.remove(proj)
            del proj


        ### Combat Mechanics ###

        self._attack_timer += dt
        if self._attack_timer < self.hit_speed:
            return True

        self._attack_timer = 0  # Reset

        for obj in self.owner.opponent.objects:
            delta = obj.position - self.position
            if delta.length() > self.attack_radius_cells:
                continue
                
            self.attack_mechanics(delta.normalize(), obj.entity_type == EntityType.AIR)
            break  # Only one arrow at a time


    def get_projectile(self, direction, target_types):
        raise NotImplementedError
    

    def attack_mechanics(self, direction, is_air_target) -> None:
        target_types = EntityType.AIR if is_air_target else set({EntityType.GROUND, EntityType.BUILDING})
        proj = self.get_projectile(direction, target_types)
        self.projectiles.append(proj)
