from game.utils import *


class Projectile:
    class TargetType:
        AREA   = 1  # Splash attack
        SINGLE = 2

    def __init__(
        self, 
        owner, 
        radius: float,

        start_position: Vector2, 
        direction: Vector2, 
        speed: float,

        projectile_type: TargetType,
        damage: int,
        max_range: float,
        target_types: Set[EntityType],
    ):
        """
        ! Design decision: This should have been a child of entity too for maintaining a lot of overall structure.
        !    But as the time of writing this, the divergence is too much to manage hence a seperate class like 
        !    this one is being written with an additional risk of mismanaged attributes and classes.
            Hence every projectile is gonna be handled differently than an entity for the seperation, and the owner
            itself manages rendering and update.

        radius: in tiles
        direction: will normalise and use this
        max_range: in tiles
        """

        self.owner = owner
        self.size = radius * 16

        self.start_position = start_position.copy()
        self.position = start_position.copy()
        self.velocity = direction.normalize() * speed

        self.projectile_type = projectile_type
        self.damage = damage
        self.max_range_cells = max_range * 16 + 32  # Add buffer for target size & frame overshoot
        self.target_types = target_types
        if not isinstance(self.target_types, set):
            self.target_types = set({self.target_types})


    def render(self, screen) -> None:
        raise NotImplementedError

    
    def update(self, dt) -> bool:
        """
        return: True means alive
        """

        if (self.position - self.start_position).length() > self.max_range_cells:
            return False

        ### Collision detection ###
        did_collide = False
        for obj in self.owner.opponent.objects:
            if not obj.has_deployed():
                continue
            if not obj.is_targetable:
                continue
            if obj.entity_type not in self.target_types:
                continue

            delta = obj.position - self.position
            if obj.entity_type == EntityType.BUILDING:
                overlap = (obj.size / 2 + Vector2(self.size, self.size)) - Vector2(abs(delta.x), abs(delta.y))

                if delta.length() > self.size + obj.size[0 if overlap.x < overlap.y else 1] / 2:
                    continue
            else:
                if delta.length() > self.size + obj.size:
                    continue

            obj.apply_damage(self.damage)
            did_collide = True

            if self.projectile_type == Projectile.TargetType.SINGLE:
                break
        
        if did_collide:
            return False


        ### Physics Update ###
        self.position += self.velocity * dt


        return True
