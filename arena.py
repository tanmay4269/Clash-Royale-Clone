from utils import *
from collections import deque

from player_side import PlayerSide, PlayerSide1, PlayerSide2
from entity import Entity

from entities.building import Building
from entities.troop import Troop

from entities.troops.knight import Knight
from entities.troops.giant import Giant
from entities.troops.mini_pekka import MiniPEKKA


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


        self.elapsed_time = 0
        self.game_duration = 180  # sec

        self.has_double_elixir_started = False
        self.double_elixir_start = 90
        

        self._font = None

        # * DEBUG *
        self._debug_active_player = 1  # For spawning the troop on the right side of the arena
        self._debug_active_card = Knight

    
    def render(self, screen, render_cell_occupancy=True) -> None:
        """        
        * For simplicity, I'll keep each sub-tile cell as one pixel
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
            (mouse_x, mouse_y) = pygame.mouse.get_pos()
            tile_x = mouse_x // self.tile_size
            tile_y = mouse_y // self.tile_size

            surface = pygame.Surface((self.tile_size, self.tile_size))
            surface.set_alpha(127)
            surface.fill((128, 128, 128))
            screen.blit(surface, (tile_x * self.tile_size, tile_y * self.tile_size))

        # HUD: elixir (bottom left) and timer (bottom right)
        if self._font is None:
            self._font = pygame.font.SysFont(None, 14)

        screen_w = self.width * self.tile_size
        screen_h = self.height * self.tile_size

        elixir_text_1 = self._font.render(f"E: {self.player_side_1.elixirs:.0f}", True, (220, 220, 220))
        screen.blit(elixir_text_1, (4, 4))

        elixir_text_2 = self._font.render(f"E: {self.player_side_2.elixirs:.0f}", True, (220, 220, 220))
        screen.blit(elixir_text_2, (4, screen_h - elixir_text_2.get_height() - 4))

        timer_text = self._font.render(f"{int(self.elapsed_time)}s", True, (220, 220, 220))
        screen.blit(timer_text, (screen_w - timer_text.get_width() - 4, 4))
 

    def update(self, dt) -> Tuple[bool, bool]:
        """
        return: 
            terminated: if someone won
            truncated: if time limit exceeded and the game hasnt terminated
        """

        self.elapsed_time += dt
        if self.elapsed_time >= self.game_duration:
            return False, True

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
                # print("Player 2 won!")
                return True, False
            elif obj == self.player_side_2.king_tower:
                # print("Player 1 won!")
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
        if not self.has_double_elixir_started and self.elapsed_time > self.double_elixir_start:
            self.has_double_elixir_started = True
            self.player_side_1.elixirs_incriment_cooldown /= 2
            self.player_side_2.elixirs_incriment_cooldown /= 2

        self.player_side_1.update(dt)
        self.player_side_2.update(dt)
        
        return False, False


    def on_click(self) -> None:
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        tile_row = mouse_y // self.tile_size
        tile_col = mouse_x // self.tile_size

        # * DEBUG * 
        owner = None
        if self._debug_active_player == 1:
            troop = self._debug_active_card(self.player_side_1, tile_row + 1, tile_col + 1)
            owner = self.player_side_1
        else:
            troop = self._debug_active_card(self.player_side_2, tile_row + 1, tile_col + 1)
            owner = self.player_side_2
        
        if self.deploy_entity(troop):
            owner.add_object(troop)


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