from game.utils import *
from game.entities.spell import Spell

@EntityRegistry.register("Arrows")
class Arrows(Spell):
    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner=owner, row=row, col=col,
            deploy_cost=3, deploy_delay=0.5,
            radius=3.5, damage=122, crown_tower_damage=31,
            **kwargs
        )
        if owner is not None:
            dist = (self.position - self.start_position).length() / 16
            # Arrows fly slightly faster than fireball (dist / 20.0)
            self.deploy_delay = max(0.5, dist / 20.0)

        self.wave_delay = 0.15
        self.max_waves = 3
        self.waves_done = 0
        
        # Override default explosion duration to cover all 3 waves plus visual fade-out
        self.explosion_duration = 0.5

        # Define 7 offsets for flying arrows cluster
        self.arrow_offsets = [
            Vector2(0, 0),
            Vector2(-15, -10),
            Vector2(12, -18),
            Vector2(-20, 15),
            Vector2(18, 12),
            Vector2(-5, -25),
            Vector2(8, 22),
        ]

    def update(self, dt, arena_cell_occupancy) -> bool:
        if self.owner is None:
            return False

        self.explosion_timer += dt

        # Apply damage waves sequentially as time progresses
        # Wave 0 triggers at 0.0s, Wave 1 at 0.15s, Wave 2 at 0.30s
        while self.waves_done < self.max_waves and self.explosion_timer >= self.waves_done * self.wave_delay:
            self.apply_spell_effect()
            self.waves_done += 1

        return self.explosion_timer < self.explosion_duration

    def render(self, screen) -> None:
        if self.owner is None:
            return

        color = "red" if self.owner.side_index == 1 else LIGHT_BLUE

        # 1. Deploy/Flying Phase
        if self._deploy_timer < self.deploy_delay:
            # Draw target circle outline on the ground
            target_surface = pygame.Surface((self.radius_cells * 2, self.radius_cells * 2), pygame.SRCALPHA)
            pygame.draw.circle(target_surface, (*pygame.Color(color)[:3], 20), (self.radius_cells, self.radius_cells), self.radius_cells)
            pygame.draw.circle(target_surface, (*pygame.Color(color)[:3], 80), (self.radius_cells, self.radius_cells), self.radius_cells, width=1)
            screen.blit(target_surface, self.position - Vector2(self.radius_cells, self.radius_cells))

            # Render the volley of flying arrows
            for i, offset in enumerate(self.arrow_offsets):
                # Stagger the progress of individual arrows in the volley
                t_i = np.clip((self._deploy_timer - i * 0.02) / (self.deploy_delay - 0.15), 0.0, 1.0)
                if t_i <= 0.0 or t_i >= 1.0:
                    continue

                target_i = self.position + offset
                current_pos_i = self.start_position + (target_i - self.start_position) * t_i
                arc_height_i = 40.0 * np.sin(t_i * np.pi)
                visual_pos_i = current_pos_i - Vector2(0, arc_height_i)

                # Estimate velocity/direction by looking slightly back in time
                prev_t_i = max(0.0, t_i - 0.04)
                prev_current_pos_i = self.start_position + (target_i - self.start_position) * prev_t_i
                prev_arc_height_i = 40.0 * np.sin(prev_t_i * np.pi)
                prev_visual_pos_i = prev_current_pos_i - Vector2(0, prev_arc_height_i)

                # Draw arrow shaft line
                pygame.draw.line(screen, (220, 220, 220), prev_visual_pos_i, visual_pos_i, 2)
                # Draw arrow tail feather (red/blue based on owner)
                pygame.draw.circle(screen, pygame.Color(color), (int(prev_visual_pos_i.x), int(prev_visual_pos_i.y)), 2)

        # 2. Active/Impact Phase
        else:
            # Render visual effects for each wave that has triggered
            for w in range(self.waves_done):
                dt_w = self.explosion_timer - (w * self.wave_delay)
                if dt_w < 0.0:
                    continue

                # Concentric expanding ripple rings
                if dt_w < 0.25:
                    ripple_pct = dt_w / 0.25
                    ripple_radius = int(self.radius_cells * ripple_pct)
                    alpha = int(180 * (1.0 - ripple_pct))
                    
                    ripple_surface = pygame.Surface((self.radius_cells * 2, self.radius_cells * 2), pygame.SRCALPHA)
                    pygame.draw.circle(ripple_surface, (*pygame.Color(color)[:3], alpha), (self.radius_cells, self.radius_cells), ripple_radius, width=2)
                    screen.blit(ripple_surface, self.position - Vector2(self.radius_cells, self.radius_cells))

                # Landing arrows sticking into the ground (fade out over 0.3s)
                if dt_w < 0.3:
                    alpha_arrows = int(255 * (1.0 - dt_w / 0.3))
                    
                    # Seed random number generator to keep arrow positions consistent per wave
                    # Use unique seed for each cast position and wave index
                    import random
                    random.seed(int(self.position.x + self.position.y * 1000 + w * 100))
                    
                    for _ in range(8):
                        # Scatter randomly in the radius circle
                        ang = random.uniform(0, 2 * np.pi)
                        rad = random.uniform(0, self.radius_cells)
                        arrow_offset = Vector2(rad * np.cos(ang), rad * np.sin(ang))
                        impact_pt = self.position + arrow_offset
                        
                        # Draw small diagonal line segment for the arrow sticking in ground
                        # Shaft
                        pygame.draw.line(
                            screen, 
                            (180, 180, 180, alpha_arrows), 
                            impact_pt - Vector2(2, 4), 
                            impact_pt, 
                            1
                        )
                        # Fletching/tail dot
                        pygame.draw.circle(
                            screen, 
                            (*pygame.Color(color)[:3], alpha_arrows), 
                            (int(impact_pt.x - 2), int(impact_pt.y - 4)), 
                            1
                        )
