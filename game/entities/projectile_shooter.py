from game.utils import *
from game.entity import Entity


class ProjectileShooter(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            if not obj.has_deployed():
                continue
            if not obj.is_targetable:
                continue
            delta = obj.position - self.position
            
            target_size = 0
            if isinstance(obj.size, Vector2):
                is_ranged = self.attack_radius_cells > 32
                if not is_ranged:
                    self_size_x = self.size.x if isinstance(self.size, Vector2) else self.size
                    self_size_y = self.size.y if isinstance(self.size, Vector2) else self.size
                    overlap = (obj.size / 2 + Vector2(self_size_x, self_size_y)) - Vector2(abs(delta.x), abs(delta.y))
                    target_size = obj.size[0 if overlap.x < overlap.y else 1] / 2
            else:
                target_size = obj.size

            if delta.length() > self.attack_radius_cells + target_size:
                continue
                
            self.launch_projectile(delta.normalize(), obj.entity_type == EntityType.AIR)
            break  # Only one arrow at a time


    def get_projectile(self, direction, target_types):
        raise NotImplementedError
    

    def launch_projectile(self, direction, is_air_target) -> None:
        target_types = EntityType.AIR if is_air_target else set({EntityType.GROUND, EntityType.BUILDING})
        proj = self.get_projectile(direction, target_types)
        self.projectiles.append(proj)
