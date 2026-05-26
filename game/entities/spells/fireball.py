from game.utils import *
from game.entities.spell import Spell

@EntityRegistry.register("Fireball")
class Fireball(Spell):
    _image_path = "assets/fireball.png"

    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner=owner, row=row, col=col,
            deploy_cost=4, deploy_delay=1.0,
            radius=2.5, damage=688, crown_tower_damage=207,
            **kwargs
        )
        if owner is not None:
            dist = (self.position - self.start_position).length() / 16
            self.deploy_delay = max(0.5, dist / 15.0)

    def render(self, screen) -> None:
        if self.owner is None:
            return

        t = np.clip(self._deploy_timer / self.deploy_delay, 0.0, 1.0)
        color = "red" if self.owner.side_index == 1 else LIGHT_BLUE

        if self._deploy_timer < self.deploy_delay:
            target_surface = pygame.Surface((self.radius_cells * 2, self.radius_cells * 2), pygame.SRCALPHA)
            pygame.draw.circle(target_surface, (*pygame.Color(color)[:3], 30), (self.radius_cells, self.radius_cells), self.radius_cells)
            pygame.draw.circle(target_surface, (*pygame.Color(color)[:3], 120), (self.radius_cells, self.radius_cells), self.radius_cells, width=1)
            screen.blit(target_surface, self.position - Vector2(self.radius_cells, self.radius_cells))

            current_pos = self.start_position + (self.position - self.start_position) * t
            arc_height = 32.0 * np.sin(t * np.pi)
            visual_pos = current_pos - Vector2(0, arc_height)

            pygame.draw.circle(screen, (40, 40, 40, 100), current_pos, 5)

            for i in range(1, 4):
                trail_t = max(0.0, t - i * 0.04)
                trail_pos = self.start_position + (self.position - self.start_position) * trail_t
                trail_arc = 32.0 * np.sin(trail_t * np.pi)
                trail_visual_pos = trail_pos - Vector2(0, trail_arc)
                trail_size = max(2, int(6 - i * 1.5))
                pygame.draw.circle(screen, (255, 100, 0, 180), trail_visual_pos, trail_size)
                pygame.draw.circle(screen, (255, 200, 0, 120), trail_visual_pos, trail_size - 1)

            pygame.draw.circle(screen, (255, 69, 0), visual_pos, 7)
            pygame.draw.circle(screen, (255, 140, 0), visual_pos, 5)
            pygame.draw.circle(screen, (255, 215, 0), visual_pos, 3)

        else:
            exp_t = np.clip(self.explosion_timer / self.explosion_duration, 0.0, 1.0)
            exp_radius = int(self.radius_cells * exp_t)
            alpha = int(200 * (1.0 - exp_t))
            
            exp_surface = pygame.Surface((self.radius_cells * 2, self.radius_cells * 2), pygame.SRCALPHA)
            pygame.draw.circle(exp_surface, (255, 69, 0, alpha), (self.radius_cells, self.radius_cells), exp_radius)
            pygame.draw.circle(exp_surface, (255, 140, 0, min(255, int(alpha * 1.2))), (self.radius_cells, self.radius_cells), int(exp_radius * 0.7))
            pygame.draw.circle(exp_surface, (255, 215, 0, alpha), (self.radius_cells, self.radius_cells), int(exp_radius * 0.4))
            
            screen.blit(exp_surface, self.position - Vector2(self.radius_cells, self.radius_cells))
