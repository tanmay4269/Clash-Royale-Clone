from game.utils import *
from game.entity import Entity
from game.entities.buildings.crown_tower import CrownTower

import heapq
from collections import deque


class Troop(Entity):
    COLLISION_COEF  = 10.0  # This times the overlap while collision is applied to this object
    FORCE_DECAY     = 0.8   # This is used to decay the acceleration per tick, assuming all forces are impulsive in the game

    class Speed:
        # TODO: units?
        SLOW   = 5.0
        MEDIUM = 10.0
        FAST   = 15.0


    class AttackRadius:
        # Units: Tiles more than the troop radius
        MELEE_SHORT  = 0.15
        MELEE_MEDIUM = 0.25  


    def __init__(
        self, owner, row, col, 
        deploy_cost, deploy_delay,
        entity_type,
        radius, 
        speed: Speed, 
        mass, 
        hitpoints,
        damage,
        attack_radius: AttackRadius,
        hit_speed, first_hit_speed,
        target_types,
        **kwargs
    ):
        """
        speed: ??   # TODO: what units are these
        mass: in kg, used to calculate acceleration by a force
        """

        super().__init__(
            owner, row, col,
            deploy_cost, deploy_delay,
            entity_type,
            hitpoints,
            damage,
            radius + attack_radius,
            hit_speed, first_hit_speed,
            target_types,
            **kwargs
        )
        
        # Physical Attributes
        self.radius = radius
        self.size = self.radius * 16
        self.position -= Vector2(self.size, self.size)  # Position adjusting to center to the cell

        self.mass = mass

        # Movement Attributes
        self.speed = speed 
        self.velocity = Vector2()
        self.acceleration = Vector2()
        
        # Navigation Attributes
        self.target = None  # Entity or None
        self._cached_target_entity = None
        self._cached_target_cell = None
        self.waypoint_reached_dist = 1.0
        self.waypoints = deque()  # left to right is the traversal pattern. self.find_path populates this and update pops from this

        self.cell_occupancy = np.zeros((int(self.size * 2), int(self.size * 2)))
        for r in range(int(self.size * 2)):
            for c in range(int(self.size * 2)):
                if (r - self.size) ** 2 + (c - self.size) ** 2 < self.size ** 3:
                    self.cell_occupancy[r, c] = 1


    def render(self, screen, color) -> None:
        if not self.has_deployed():
            smooth_arc(screen, "black", 
                pygame.Rect(
                    self.position.x - self.size - 2,
                    self.position.y - self.size - 2,
                    self.size * 2 + 4, 
                    self.size * 2 + 4, 
                ),
                start_angle= -np.pi/2,
                stop_angle= -np.pi/2 + 2 * np.pi * np.clip(self._deploy_timer / self.deploy_delay, 0.0, 1.0),
                width=2
            )

        # Body
        pygame.draw.circle(screen, color, self.position, self.size)
        pygame.draw.circle(screen, "black", self.position, self.size, width=1)

        # Health Bar
        health_bar_length = 30 * (self.health / self.hitpoints)

        shape = (
            self.position.x - 15,
            self.position.y - self.size - 15,
            health_bar_length,
            4
        )

        pygame.draw.rect(screen, color, shape)
        pygame.draw.rect(screen, "black", shape, width=1)

        ### * Debug * ###

        # if False:
        if True:
            # Attack radius
            # pygame.draw.circle(screen, "black", self.position, self.attack_radius_cells, width=1)
            
            # Visibility radius
            # pygame.draw.circle(screen, "brown", self.position, self.visibility_cells, width=1)

            # Waypoints
            for i in range(len(self.waypoints)-1):
                pygame.draw.line(screen, "green", self.waypoints[i], self.waypoints[i+1], width=2)


    def update(self, dt, arena_cell_occupancy) -> bool:
        """
        If target is not None, pathfind to that and take incremental steps towards it
        Else set target to the closest compatable victim 
        """

        if self.health < 0:
            return False


        ### Navigation ###
        self.velocity = Vector2()  # Reset velocity each tick; TODO Think this through

        # Every tick, update target and path to it
        #   TODO Wasteful approach, need to do something smarter
        prev_target = self.target
        self.set_target()
        
        if self.target is None:
            return True

        if self.target != prev_target:
            self._is_first_hit = True
            self._attack_timer = 0  # Reset timer to discard stale delay from last attack
        

        ### If the target is in reach, navigate to it ###

        # Get target size for better estimation of proximity
        target_size = 0
        if isinstance(self.target.size, Vector2):
            # ! Hacky way to know if its a rectangular target
            #   but its established that only rects and circles are gonna be there 
            #   and buildings are only rects and troops are just circles
            delta = self.target.position - self.position
            overlap = (self.target.size / 2 + Vector2(self.size, self.size)) - Vector2(abs(delta.x), abs(delta.y))

            target_size = self.target.size[0 if overlap.x < overlap.y else 1] / 2
        else:
            target_size = self.target.size

        # Check target proximity
        is_ranged = self.attack_radius_cells > 32
        eff_target_size = 0 if (is_ranged and isinstance(self.target.size, Vector2)) else target_size

        found_path = False
        if (self.position - self.target.position).length() > self.attack_radius_cells + eff_target_size:
            found_path = self.find_path(arena_cell_occupancy)

            if not found_path:
                print("(WARN: Troop::update): Couldn't find a path")

        if found_path:
            if len(self.waypoints) == 0:
                self.target = None
                return True
            
            displacement = self.waypoints[0] - self.position

            if displacement.length() < self.waypoint_reached_dist:
                self.waypoints.popleft()

            self.velocity = displacement.normalize() * self.speed
        

        ### Physics Update ###
        self.velocity += dt * self.acceleration
        self.position += dt * self.velocity

        self.acceleration *= Troop.FORCE_DECAY  # * Assuming all forces are impulsive in the game
        

        ### Combat Mechanics ###
        if (self.position - self.target.position).length() < self.attack_radius_cells + eff_target_size: 
            self.attack_mechanics() 
            self._attack_timer += dt


        return True


    ############################
    ### Navigation Mechanics ###
    ############################

    def set_target(self, target=None):
        """
        If no target is given, find a target using self.owner.opponent.objects.

        Tower targeting rules (half-based, Clash Royale):
          - The arena is split into left (x < ARENA_MID_X) and right halves.
          - A troop's half is determined by its current x-position.
          - Default navigation tower = same-half enemy princess tower if alive,
            otherwise the king tower.
          - Non-tower targets (troops / buildings) within visibility range always
            take priority over the default tower.
        """
        from game.entities.buildings.king_tower import KingTower as KingTowerClass
        from game.entities.buildings.princess_tower import PrincessTower as PrincessTowerClass

        # Doesn't need to know who the target is, just knowing the location is fine
        if target:
            self.target = target
            return

        opponent = self.owner.opponent

        # --- Half-based default tower selection ---
        # Arena is 18 tiles wide, tile_size=16 px → midline at x = 9*16 = 144 px.
        ARENA_MID_X = 9 * 16  # pixels

        on_left_half = self.position.x < ARENA_MID_X

        # princess_tower_1 is on the left (col 3.5), princess_tower_2 on the right (col 14.5)
        same_half_princess = (
            opponent.princess_tower_1 if on_left_half else opponent.princess_tower_2
        )

        # Check if that princess tower is still alive (present in opponent.objects) and deployed
        if (
            same_half_princess is not None
            and same_half_princess in opponent.objects
            and same_half_princess.has_deployed()
            and same_half_princess.entity_type in self.target_types
        ):
            default_tower = same_half_princess
        else:
            # Same-half princess tower is gone → head straight for the king tower
            default_tower = opponent.king_tower if (
                opponent.king_tower is not None
                and opponent.king_tower.has_deployed()
                and opponent.king_tower.entity_type in self.target_types
            ) else None

        # --- Scan for a closer non-tower target (troops / buildings) in range ---
        closest_obj, closest_dist = None, float('inf')
        for obj in opponent.objects:
            if not obj.has_deployed():
                continue
            if not obj.is_targetable:
                continue
            if obj.entity_type not in self.target_types:
                continue

            # Crown towers are handled via default_tower above; skip them in the
            # proximity scan so a nearby king tower doesn't steal priority.
            if isinstance(obj, CrownTower):
                continue

            dist = (self.position - obj.position).length()
            if dist > self.visibility_cells:
                continue

            if dist < closest_dist:
                closest_obj = obj
                closest_dist = dist

        # Use the closest in-range non-tower target if found, otherwise fall back
        # to the default navigation tower.
        self.target = closest_obj if closest_obj is not None else default_tower


    def find_path(self, occupancy_grid: np.ndarray) -> bool:
        """
        Given curent position and target position, this computes the right sequence
        of waypoints that this entity needs to navigate through.

        Returns False if path couldn't be found for some reason
        """

        if self.target is None:
            return False

        SCALE = 16  # Reduction by this much on each axis

        # Center-pixel sample for walls (avoids bleed into adjacent tiles).
        center = occupancy_grid[SCALE//2::SCALE, SCALE//2::SCALE]

        # Max-pool across the full tile only for the building layer:
        # if a building covers even one sub-pixel of a tile, block the whole tile.
        h, w = occupancy_grid.shape
        building_layer = (occupancy_grid == 2).astype(np.uint8)
        building_any = (
            building_layer
            .reshape(h // SCALE, SCALE, w // SCALE, SCALE)
            .max(axis=(1, 3))
        ).astype(bool)

        # Block: permanent walls (center-pixel) OR buildings (max-pool); troops passable.
        if self.entity_type == EntityType.AIR:
            # Air troops fly over river (center == 1)
            tiled_occupancy_grid = np.where(
                building_any, 1, 0
            ).astype(int)
        else:
            tiled_occupancy_grid = np.where(
                (center == 1) | building_any, 1, 0
            ).astype(int)

        grid_rows, grid_cols = tiled_occupancy_grid.shape

        # All free tile indices, computed once and shared by both nearest_free calls.
        free_cells = np.argwhere(tiled_occupancy_grid == 0)  # shape (K, 2)

        from game.entities.building import Building

        def nearest_free_entity(entity, toward_entity=None):
            """Snap an entity's position to the nearest unblocked cell.
            Uses bounding-box distance for buildings, and center tile distance for troops."""
            tx = int(np.clip(entity.position.x / SCALE, 0, grid_rows - 1))
            ty = int(np.clip(entity.position.y / SCALE, 0, grid_cols - 1))
            if tiled_occupancy_grid[tx, ty] == 0:
                return (tx, ty)

            if isinstance(entity, Building):
                left = entity.position.x - entity.size.x / 2
                right = entity.position.x + entity.size.x / 2
                top = entity.position.y - entity.size.y / 2
                bottom = entity.position.y + entity.size.y / 2

                r_min = int(np.floor(left / SCALE))
                r_max = int(np.ceil(right / SCALE)) - 1
                c_min = int(np.floor(top / SCALE))
                c_max = int(np.ceil(bottom / SCALE)) - 1
            else:
                r_min, r_max, c_min, c_max = tx, tx, ty, ty

            r_min = max(0, min(r_min, grid_rows - 1))
            r_max = max(0, min(r_max, grid_rows - 1))
            c_min = max(0, min(c_min, grid_cols - 1))
            c_max = max(0, min(c_max, grid_cols - 1))

            dr = np.maximum(0, np.maximum(r_min - free_cells[:, 0], free_cells[:, 0] - r_max))
            dc = np.maximum(0, np.maximum(c_min - free_cells[:, 1], free_cells[:, 1] - c_max))
            d2 = dr ** 2 + dc ** 2
            min_d2 = d2.min()

            candidates = free_cells[d2 == min_d2]
            if toward_entity is not None and len(candidates) > 1:
                ref_r = toward_entity.position.x / SCALE
                ref_c = toward_entity.position.y / SCALE
                d_toward = (candidates[:, 0] - ref_r) ** 2 + (candidates[:, 1] - ref_c) ** 2
                best = candidates[np.argmin(d_toward)]
            else:
                best = candidates[0]
            return (int(best[0]), int(best[1]))

        # Start nudge toward target, target nudge toward troop
        start = nearest_free_entity(self, self.target)

        # Cache the target snapped cell to avoid oscillations as the troop moves
        cached_cell = getattr(self, '_cached_target_cell', None)
        cached_ent = getattr(self, '_cached_target_entity', None)
        if (
            cached_ent is None
            or cached_ent != self.target
            or cached_cell is None
            or tiled_occupancy_grid[cached_cell[0], cached_cell[1]] != 0
        ):
            self._cached_target_entity = self.target
            self._cached_target_cell = nearest_free_entity(self.target, self)

        target = self._cached_target_cell

        path = self.a_star(tiled_occupancy_grid, start, target)

        if path is None:
            return False

        self.waypoints = deque()
        for wp in path[1:-1]:
            self.waypoints.append(Vector2(wp) * SCALE + Vector2(SCALE/2, SCALE/2))
        # Last waypoint: actual target position for a natural approach
        self.waypoints.append(self.target.position.copy())

        return True


    def a_star(self, grid, start, goal):
        """
        8-way connected on the grid
        Uses Octile Distance huristic
        """

        rows, cols = grid.shape

        def heuristic(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)  # Octile

        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                    (-1,-1),(-1,1),(1,-1),(1,1)]

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return [start] + path[::-1]

            for dr, dc in neighbors:
                neighbor = (current[0]+dr, current[1]+dc)
                r, c = neighbor
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                # Allow stepping onto the goal even if it's "occupied" by a building
                if grid[r, c] != 0 and neighbor != goal:
                    continue

                # Diagonal moves cost sqrt(2), cardinal cost 1
                move_cost = np.sqrt(2) if dr != 0 and dc != 0 else 1
                new_g = g_score[current] + move_cost

                if new_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None  # No path found


    def get_cell_occupancy_index(self):
        return 3


    def get_cell_occupancy(self):
        return np.ones([1, 1]) * self.get_cell_occupancy_index(), self.position - Vector2(self.size, self.size)
        # return self.cell_occupancy * self.get_cell_occupancy_index(), self.position - Vector2(self.size, self.size)


    def apply_force(self, force: Vector2) -> None:
        self.acceleration += force / self.mass

        # Hacky -- Doesn't look great
        # self.position += force.normalize() * 0.5


    ########################
    ### Combat Mechanics ###
    ########################

    def attack_mechanics(self) -> None:
        """
        target is reused from self.target
        """

        if self._is_first_hit and self._attack_timer < self.first_hit_speed:
            return
        elif not self._is_first_hit and self._attack_timer < self.hit_speed:
            return

        self._is_first_hit = False
        self._attack_timer = 0  # Reset
        self.target.apply_damage(self.damage)
