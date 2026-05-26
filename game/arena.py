from game.utils import *
from collections import deque

from game.player_side import PlayerSide, PlayerSide1, PlayerSide2
from game.entity import Entity

from game.entities.building import Building
from game.entities.troop import Troop

from game.entities.troops.knight import Knight
from game.entities.troops.giant import Giant
from game.entities.troops.mini_pekka import MiniPEKKA


class Arena:
    def __init__(self):
        self.tile_size = 16  # Sub-tile cells
            # * Each cell is just one pixel as off now, temporary simplificaton

        self.width = 18  # In tiles
        self.height = 32

        self.max_num_objects = 32  # Arbitrary
        self.objects: List[Entity] = []  # Contains all in game objects that have been deployed like buildings, troupes, etc.
        self.deploy_buffer: List[Entity] = []  # Contains items that haven't deployed yet but need to be rendered

        # Occupancy Grid
        #   0 => Unoccupied
        #   1 => Permanent occupancy
        #   2 => Building
        #   3 => Troop
        # TODO: This should be its own class coz I later also wanna implement an `on_update`
        #   that depends on this matrix being updated and only upon update down stream recomputation occurs
        self.cell_occupancy = np.zeros(
            (self.width * self.tile_size, 
            self.height * self.tile_size), 
        )

        # Top and bottom rows
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (0, 0))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (self.tile_size * 12, 0))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (0, (self.height-1) * self.tile_size))
        self.occupy_cells(np.ones((self.tile_size * 6, self.tile_size)), (self.tile_size * 12, (self.height-1) * self.tile_size))

        # The divider
        self.occupy_cells(np.ones((self.tile_size * 5//2, self.tile_size * 2)), (0, self.tile_size * 15))
        self.occupy_cells(np.ones((self.tile_size * 5//2, self.tile_size * 2)), (self.tile_size * 31//2, self.tile_size * 15))
        self.occupy_cells(np.ones((self.tile_size * 9, self.tile_size * 2)),    (self.tile_size * 9//2, self.tile_size * 15))
        
        # Adding player sides
        self.player_side_1 = PlayerSide1()  # The one closer to (0, 0)
        self.player_side_2 = PlayerSide2()

        self.player_side_1.set_opponent(self.player_side_2)
        self.player_side_2.set_opponent(self.player_side_1)

        # Deploying crown towers: use list concat to preserve deterministic order
        towers = list(self.player_side_1.get_objects()) + list(self.player_side_2.get_objects())
        for obj in towers:
            self.deploy_entity(obj)

        # 
        self.elapsed_time = 0
        self.game_duration = 300  # 5:00 — extended for sudden death

        self.has_double_elixir_started = False
        self.has_triple_elixir_started = False
        self.has_sudden_death_started = False
        self.double_elixir_start = 120   # 2:00 mark
        self.triple_elixir_start = 180   # 3:00 mark
        self.sudden_death_start = 180    # 3:00 — same tick as triple elixir

        # Set by update() when the game ends; 1 or 2 for the winning player, None otherwise
        self.winner: int | None = None
        
        # 
        self._font = None

        # * DEBUG *
        self._debug_active_player = 1  # For spawning the troop on the right side of the arena
        self._debug_active_card = Knight

    
    def render(self, screen, render_cell_occupancy=True) -> None:
        """        
        For simplicity, I'll keep each sub-tile cell as one pixel
        """

        # Ground layer
        screen.fill("#D1CC95")

        # Occupancy map
        if render_cell_occupancy:
            # TODO: Make it so that this isn't wastefuly recomputed each time
            rgb_occupancy_map = np.empty((self.cell_occupancy.shape[0], self.cell_occupancy.shape[1], 3), dtype=np.uint8)

            # TODO: Manage multiple layers
            rgb_occupancy_map[:, :, 0] = 255 * (1 - self.cell_occupancy)
            rgb_occupancy_map[:, :, 1] = 255 * (1 - self.cell_occupancy)
            rgb_occupancy_map[:, :, 2] = 255 * (1 - self.cell_occupancy)

            surface = pygame.surfarray.make_surface(rgb_occupancy_map)
            surface.set_alpha(127)
            screen.blit(surface, (0, 0))


        # Faint gridlines 
        #   TODO: This can probably be optimised by precomputing a sprite 
        for r in range(self.height):
            for c in range(self.width):
                pygame.draw.line(screen, (128, 128, 128), (0, self.tile_size * r), (self.tile_size * self.width, self.tile_size * r), 1)
                pygame.draw.line(screen, (128, 128, 128), (self.tile_size * c, 0), (self.tile_size * c, self.tile_size * self.height), 1)


        # All objects
        for obj in self.objects:
            obj.render(screen)

        for obj in self.deploy_buffer:
            obj.render(screen)

        # Highlighted cell under the cursor
        if pygame.display.get_init():
            display_surf = pygame.display.get_surface()
            if display_surf is not None:
                (m_x, m_y) = pygame.mouse.get_pos()
                scale = display_surf.get_width() / screen.get_width()
                mouse_x = int(m_x / scale)
                mouse_y = int(m_y / scale)
                tile_x = mouse_x // self.tile_size
                tile_y = mouse_y // self.tile_size

                if 0 <= tile_x < self.width and 0 <= tile_y < self.height:
                    surface = pygame.Surface((self.tile_size, self.tile_size))
                    surface.set_alpha(127)
                    surface.fill((128, 128, 128))
                    screen.blit(surface, (tile_x * self.tile_size, tile_y * self.tile_size))

        # HUD: mode indicator (top left) | E: X  MM:SS (top right) | E: X (bottom left)
        if self._font is None:
            self._font = pygame.font.SysFont(None, 14)

        screen_w = self.width * self.tile_size
        screen_h = self.height * self.tile_size

        # Count-up timer in MM:SS — top right
        elapsed = int(self.elapsed_time)
        mins = elapsed // 60
        secs = elapsed % 60
        timer_text = self._font.render(f"{mins}:{secs:02d}", True, (220, 220, 220))
        timer_x = screen_w - timer_text.get_width() - 4
        screen.blit(timer_text, (timer_x, 4))

        # Elixir / Sudden Death mode indicator — top right, just to the left of the timer
        if self.has_sudden_death_started:
            mode_label = "Sudden Death!"
            mode_color = (255, 50, 50)    # bright red
        elif self.has_triple_elixir_started:
            mode_label = "3x Elixir!"
            mode_color = (255, 100, 50)   # orange-red
        elif self.has_double_elixir_started:
            mode_label = "2x Elixir!"
            mode_color = (100, 220, 255)  # cyan-blue
        else:
            mode_label = None

        if mode_label is not None:
            mode_text = self._font.render(mode_label, True, mode_color)
            screen.blit(mode_text, (timer_x - mode_text.get_width() - 6, 4))

        # Player 1 elixir — top left
        elixir_text_1 = self._font.render(f"E: {self.player_side_1.elixirs:.0f}", True, (220, 220, 220))
        screen.blit(elixir_text_1, (4, 4))

        # Player 2 elixir — bottom left
        elixir_text_2 = self._font.render(f"E: {self.player_side_2.elixirs:.0f}", True, (220, 220, 220))
        screen.blit(elixir_text_2, (4, screen_h - elixir_text_2.get_height() - 4))

        # Draw the right side deck/hand HUD panel
        self.draw_hud_panel(screen)


    def draw_hud_panel(self, screen) -> None:
        # Draw panel background
        panel_rect = pygame.Rect(self.width * self.tile_size, 0, 200, self.height * self.tile_size)
        pygame.draw.rect(screen, (24, 24, 28), panel_rect)
        pygame.draw.line(screen, (80, 80, 80), (self.width * self.tile_size, 0), (self.width * self.tile_size, self.height * self.tile_size), 2)

        # Card metadata
        CARD_INFO = {
            "Knight": {"char": "K", "cost": 3, "color": (0, 180, 255)},
            "Giant": {"char": "G", "cost": 5, "color": (230, 160, 0)},
            "MiniPEKKA": {"char": "P", "cost": 4, "color": (220, 40, 40)},
            "Musketeer": {"char": "M", "cost": 4, "color": (160, 80, 220)},
            "Archers": {"char": "A", "cost": 3, "color": (40, 200, 80)},
            "Fireball": {"char": "F", "cost": 4, "color": (255, 69, 0)},
        }

        # Fonts
        title_font = pygame.font.SysFont(None, 16, bold=True)
        small_font = pygame.font.SysFont(None, 12)
        badge_font = pygame.font.SysFont(None, 10, bold=True)
        
        # Draw Player 1 HUD (Human, top half)
        self._draw_player_hud_block(screen, self.player_side_1, 1, 20, CARD_INFO, title_font, small_font, badge_font)

        # Draw Player 2 HUD (Opponent, bottom half)
        self._draw_player_hud_block(screen, self.player_side_2, 2, 320, CARD_INFO, title_font, small_font, badge_font)


    def _draw_player_hud_block(self, screen, player, player_idx, y_offset, card_info, title_font, small_font, badge_font):
        # 1. Header Text
        title_text = f"PLAYER {player_idx}"
        is_active = (self._debug_active_player == player_idx)
        title_color = (255, 215, 0) if is_active else (180, 180, 180)
        title_surface = title_font.render(title_text, True, title_color)
        screen.blit(title_surface, (295, y_offset))

        # Active indicator bullet
        if is_active:
            pygame.draw.circle(screen, (0, 255, 100), (370, y_offset + 6), 4)

        # 2. Elixir Text & Progress Bar
        elixir_val = player.elixirs
        elixir_text = f"Elixir: {elixir_val:.1f}"
        elixir_surface = small_font.render(elixir_text, True, (200, 200, 200))
        screen.blit(elixir_surface, (295, y_offset + 18))

        # Elixir Bar background
        bar_x, bar_y, bar_w, bar_h = 295, y_offset + 32, 180, 8
        pygame.draw.rect(screen, (40, 20, 50), (bar_x, bar_y, bar_w, bar_h), border_radius=3)

        # Elixir Bar fill
        fill_w = int(bar_w * (elixir_val / player.max_elixirs))
        if fill_w > 0:
            pygame.draw.rect(screen, (220, 50, 220), (bar_x, bar_y, fill_w, bar_h), border_radius=3)
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_w, bar_h), width=1, border_radius=3)

        # 3. 4 Hand Cards
        card_y = y_offset + 48
        for i, card_cls in enumerate(player.hand):
            card_name = card_cls.__name__
            info = card_info.get(card_name, {"char": "?", "cost": 0, "color": (128, 128, 128)})
            char = info["char"]
            cost = info["cost"]
            color = info["color"]

            card_x = 295 + i * 36
            card_w, card_h = 30, 42
            card_rect = pygame.Rect(card_x, card_y, card_w, card_h)

            # Check if active selection
            is_selected = (player.active_card_idx == i)

            # Draw card slot background
            bg_color = (40, 40, 45) if elixir_val < cost else (55, 55, 65)
            pygame.draw.rect(screen, bg_color, card_rect, border_radius=4)

            # Elixir fill animation
            if elixir_val < cost:
                fill_fraction = elixir_val / cost
                fill_height = int(card_h * fill_fraction)
                if fill_height > 0:
                    # Draw purple fill from bottom
                    fill_rect = pygame.Rect(card_x, card_y + card_h - fill_height, card_w, fill_height)
                    pygame.draw.rect(screen, (120, 40, 150), fill_rect, border_radius=4)
            else:
                # Fully charged card background highlight
                # Draw a subtle tint of the card's theme color
                tint_surface = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                tint_surface.fill((*color, 40)) # 40/255 opacity tint
                screen.blit(tint_surface, (card_x, card_y))

            # Selected Border (Glowing)
            if is_selected:
                pygame.draw.rect(screen, (0, 255, 255), (card_x - 1, card_y - 1, card_w + 2, card_h + 2), width=2, border_radius=4)
            else:
                pygame.draw.rect(screen, (100, 100, 100), card_rect, width=1, border_radius=4)

            # Card Letter
            letter_surface = title_font.render(char, True, color)
            letter_x = card_x + (card_w - letter_surface.get_width()) // 2
            letter_y = card_y + (card_h - letter_surface.get_height()) // 2
            screen.blit(letter_surface, (letter_x, letter_y))

            # Cost Badge in top-right
            badge_rect = pygame.Rect(card_x + card_w - 9, card_y - 3, 12, 12)
            pygame.draw.circle(screen, (120, 20, 180), badge_rect.center, 6)
            cost_surface = badge_font.render(str(cost), True, (255, 255, 255))
            cost_x = badge_rect.center[0] - cost_surface.get_width() // 2
            cost_y = badge_rect.center[1] - cost_surface.get_height() // 2
            screen.blit(cost_surface, (cost_x, cost_y))

        # 4. Next Card Slot
        next_x = 295 + 4 * 36 + 6
        next_y = card_y + 6
        next_w, next_h = 22, 30
        next_rect = pygame.Rect(next_x, next_y, next_w, next_h)

        # Label "NEXT"
        next_label = small_font.render("NEXT", True, (150, 150, 150))
        screen.blit(next_label, (next_x - 2, next_y - 12))

        # Draw Next Card slot background
        pygame.draw.rect(screen, (30, 30, 35), next_rect, border_radius=3)
        pygame.draw.rect(screen, (70, 70, 70), next_rect, width=1, border_radius=3)

        if player.next_card:
            next_name = player.next_card.__name__
            info = card_info.get(next_name, {"char": "?", "cost": 0, "color": (128, 128, 128)})
            char = info["char"]
            color = info["color"]

            next_letter_surface = small_font.render(char, True, color)
            next_letter_x = next_x + (next_w - next_letter_surface.get_width()) // 2
            next_letter_y = next_y + (next_h - next_letter_surface.get_height()) // 2
            screen.blit(next_letter_surface, (next_letter_x, next_letter_y))


    def update(self, dt) -> Tuple[bool, bool]:
        """
        return: 
            terminated: if someone won
            truncated: if time limit exceeded and the game hasnt terminated
        """

        self.elapsed_time += dt
        if self.elapsed_time >= self.game_duration:
            self._resolve_tiebreaker()
            return False, True

        in_sudden_death = self.has_sudden_death_started

        ### Collision Management ###

        # Makes a reasonable simplifying assumption that buildings are rects and troops are circles
        # TODO: Maybe much later I implement spacial proximity based approach. If things lag, this could be an optimisation
        # TODO: Put this in another method
        for i, obj_i in enumerate(self.objects):
            for j in range(i + 1, len(self.objects)):
                obj_j = self.objects[j]

                # Troop-Troop Collision
                if isinstance(obj_i, Troop) and isinstance(obj_j, Troop):
                    dx = obj_i.position.x - obj_j.position.x
                    dy = obj_i.position.y - obj_j.position.y
                    dist_sq = dx**2 + dy**2
                    rad_sum = obj_i.size + obj_j.size
                    
                    if dist_sq >= rad_sum**2:
                        continue
                        
                    dist = dist_sq**0.5
                    overlap = rad_sum - dist
                    if dist_sq < 1e-6:
                        dx, dy, dist = 0.1, 0.1, 0.1414
                        
                    fx = (dx / dist) * overlap * Troop.COLLISION_COEF
                    fy = (dy / dist) * overlap * Troop.COLLISION_COEF
                    
                    obj_i.apply_force(Vector2(fx, fy))
                    obj_j.apply_force(Vector2(-fx, -fy))

                # Building-Troop Collision
                elif isinstance(obj_i, Building) and isinstance(obj_j, Troop):
                    dx = obj_i.position.x - obj_j.position.x
                    dy = obj_i.position.y - obj_j.position.y
                    ox = (obj_i.size.x / 2 + obj_j.size) - abs(dx)
                    oy = (obj_i.size.y / 2 + obj_j.size) - abs(dy)
                    
                    if ox < 0 or oy < 0:
                        continue
                        
                    dist_sq = dx**2 + dy**2
                    dist = dist_sq**0.5 if dist_sq >= 1e-6 else 0.1414
                    dx, dy = (dx, dy) if dist_sq >= 1e-6 else (0.1, 0.1)
                        
                    overlap_len = (ox**2 + oy**2)**0.5
                    fx = -(dx / dist) * overlap_len * Troop.COLLISION_COEF
                    fy = -(dy / dist) * overlap_len * Troop.COLLISION_COEF
                    obj_j.apply_force(Vector2(fx, fy))
                    
                # Troop-Building Collision (swapped)
                elif isinstance(obj_i, Troop) and isinstance(obj_j, Building):
                    dx = obj_j.position.x - obj_i.position.x
                    dy = obj_j.position.y - obj_i.position.y
                    ox = (obj_j.size.x / 2 + obj_i.size) - abs(dx)
                    oy = (obj_j.size.y / 2 + obj_i.size) - abs(dy)
                    
                    if ox < 0 or oy < 0:
                        continue
                        
                    dist_sq = dx**2 + dy**2
                    dist = dist_sq**0.5 if dist_sq >= 1e-6 else 0.1414
                    dx, dy = (dx, dy) if dist_sq >= 1e-6 else (0.1, 0.1)
                        
                    overlap_len = (ox**2 + oy**2)**0.5
                    fx = -(dx / dist) * overlap_len * Troop.COLLISION_COEF
                    fy = -(dy / dist) * overlap_len * Troop.COLLISION_COEF
                    obj_i.apply_force(Vector2(fx, fy))


        ### Deploy Buffer Management ###

        # Snapshot which objects have finished deploying (list comprehension = deterministic order)
        deployed_objs = [obj for obj in self.deploy_buffer if obj.has_deployed(dt)]
        for obj in deployed_objs:
            if len(self.objects) < self.max_num_objects:
                self.objects.append(obj)
                # ! ADD TO PLAYER OBJECTS TOO
            else:
                print("(WARN: Arena::update) Buffer Management: Can't deploy since max num objects has been reached")
            self.deploy_buffer.remove(obj)


        ### Object Update and Deletion ###

        # Snapshot dead objects first (list comprehension = deterministic order),
        # then delete: avoids mutating self.objects while iterating it
        dead_objs = [obj for obj in self.objects if not obj.update(dt, self.cell_occupancy)]
        for obj in dead_objs:
            if obj.owner.side_index == 1:
                self.player_side_1.remove_object(obj)
            elif obj.owner.side_index == 2:
                self.player_side_2.remove_object(obj)

            if obj == self.player_side_1.king_tower:
                # King tower destroyed → opponent wins
                self.winner = 2
                return True, False
            elif obj == self.player_side_2.king_tower:
                self.winner = 1
                return True, False

            # During sudden death any tower kill ends the game immediately
            if in_sudden_death and isinstance(obj, Building):
                if obj == self.player_side_1.princess_tower_1 or \
                   obj == self.player_side_1.princess_tower_2:
                    # P1's princess tower killed → P2 wins
                    self.winner = 2
                    # Still clear the footprint before returning
                    mask, mask_pos = obj.get_cell_occupancy()
                    self.occupy_cells(np.zeros_like(mask), mask_pos)
                    self.objects.remove(obj)
                    return True, False
                elif obj == self.player_side_2.princess_tower_1 or \
                     obj == self.player_side_2.princess_tower_2:
                    # P2's princess tower killed → P1 wins
                    self.winner = 1
                    mask, mask_pos = obj.get_cell_occupancy()
                    self.occupy_cells(np.zeros_like(mask), mask_pos)
                    self.objects.remove(obj)
                    return True, False

            # Clear the dead building's footprint from the occupancy grid so
            # troops can navigate through the space it used to occupy.
            if isinstance(obj, Building):
                mask, mask_pos = obj.get_cell_occupancy()
                clear_mask = np.zeros_like(mask)
                self.occupy_cells(clear_mask, mask_pos)

            self.objects.remove(obj)
            # del obj


        ### Elixir Update ###
        if not self.has_double_elixir_started and self.elapsed_time >= self.double_elixir_start:
            self.has_double_elixir_started = True
            self.player_side_1.set_double_elixir_mode()
            self.player_side_2.set_double_elixir_mode()

        if not self.has_triple_elixir_started and self.elapsed_time >= self.triple_elixir_start:
            self.has_triple_elixir_started = True
            self.player_side_1.set_tripple_elixir_mode()
            self.player_side_2.set_tripple_elixir_mode()

        ### Sudden Death ###
        if not self.has_sudden_death_started and self.elapsed_time >= self.sudden_death_start:
            self.has_sudden_death_started = True

        self.player_side_1.update(dt)
        self.player_side_2.update(dt)
        
        return False, False


    def _resolve_tiebreaker(self) -> None:
        """
        Called when the 5:00 hard limit is hit.
        Rule 1: most living towers wins.
        Rule 2: if equal towers, least total remaining HP loses (more damage taken).
        Sets self.winner to 1, 2, or leaves it None on a true draw.
        """
        def _living_towers(side):
            towers = [side.king_tower, side.princess_tower_1, side.princess_tower_2]
            return [t for t in towers if t in self.objects]

        p1_towers = _living_towers(self.player_side_1)
        p2_towers = _living_towers(self.player_side_2)

        n1, n2 = len(p1_towers), len(p2_towers)

        if n1 > n2:
            self.winner = 1
        elif n2 > n1:
            self.winner = 2
        else:
            # Equal tower count → compare total remaining HP
            hp1 = sum(t.health for t in p1_towers)
            hp2 = sum(t.health for t in p2_towers)
            if hp1 > hp2:
                self.winner = 1   # P1 has more HP remaining → P2 took more damage → P1 wins
            elif hp2 > hp1:
                self.winner = 2
            # else: true draw, winner stays None
    

    def on_click(self, mouse_pos=None) -> None:
        if mouse_pos is None:
            (m_x, m_y) = pygame.mouse.get_pos()
            display_surf = pygame.display.get_surface()
            if display_surf is not None:
                scale = display_surf.get_width() / (self.width * self.tile_size + 200)
                mouse_x = int(m_x / scale)
                mouse_y = int(m_y / scale)
            else:
                mouse_x, mouse_y = m_x, m_y
        else:
            (mouse_x, mouse_y) = mouse_pos
        tile_row = mouse_y // self.tile_size
        tile_col = mouse_x // self.tile_size
        owner = self.player_side_1 if self._debug_active_player == 1 else self.player_side_2
        if owner.active_card_idx is None:
            return

        card_cls = owner.hand[owner.active_card_idx]
        troop = card_cls(owner, tile_row + 1, tile_col + 1)

        if self.deploy_entity(troop):
            owner.add_object(troop)
            owner.use_card(owner.active_card_idx)


    def deploy_entity(self, deploy_me: Entity) -> bool:
        """
        Return false if the entity can't be deployed in its current form
        """

        # 1. Check if player has enough elixir
        if deploy_me.owner.elixirs < deploy_me.deploy_cost:
            return False
        
        # 2. Check if the deploy location (already written into the object, 
        #   access via public method) is available to deploy, if not return False
        mask, mask_pos = deploy_me.get_cell_occupancy()
        if self.occupy_cells(mask, mask_pos) is False:
            return False

        # 3. Add to deploy buffer
        if hasattr(deploy_me, "get_units"):
            for unit in deploy_me.get_units():
                self.deploy_buffer.append(unit)
        else:
            self.deploy_buffer.append(deploy_me)

        # 4. Subtract player's elixirs and return True
        deploy_me.owner.spend_elixirs(deploy_me.deploy_cost)

        return True


    def occupy_cells(self, mask: np.ndarray, mask_pos) -> bool:
        """
        If mask overlaps with something, return false
        mask_pos is expected to be x, y and mask is to be width, height shaped
        
        Can also be used to "unoccupy" cells
        """

        if isinstance(mask_pos, Vector2):
            mask_pos = (int(mask_pos.x), int(mask_pos.y))

        # 1. Check with self.cell_occupancy, if any intersection, return false
        row_min, row_max = mask_pos[0], mask_pos[0] + mask.shape[0]
        col_min, col_max = mask_pos[1], mask_pos[1] + mask.shape[1]

        tmp_mask = self.cell_occupancy[
            row_min : row_max, 
            col_min : col_max, 
        ]

        # Check on each layer
        for bg_layer in range(1, Entity.CELL_OCCUPANCY_LAYERS+1):
            for fg_layer in range(bg_layer, Entity.CELL_OCCUPANCY_LAYERS+1):
                if bg_layer == 3 and bg_layer == fg_layer:
                    continue   # Don't check for troop-troop deployment constraint
                tmp_mask_layer  = np.where(tmp_mask == bg_layer, True, False)
                mask_layer      = np.where(mask == fg_layer, True, False)

                if np.any(tmp_mask_layer & mask_layer):
                    return False

        # 2. Else just OR it to the cell occupancy
        self.cell_occupancy[
            row_min : row_max, 
            col_min : col_max, 
        ] = mask

        return True
