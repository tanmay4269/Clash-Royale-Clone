from game.utils import *
from game.entity import Entity


class Building(Entity):
    def __init__(
        self, owner, row: int, col: int, 
        deploy_cost: int,
        deploy_delay: float,
        entity_type,
        width: int, height: int,
        hitpoints,
        damage,
        attack_radius,
        hit_speed, first_hit_speed,
        target_types,
        **kwargs
    ):
        super().__init__(
            owner=owner, row=row, col=col,
            deploy_cost=deploy_cost, deploy_delay=deploy_delay,
            entity_type=entity_type,
            hitpoints=hitpoints,
            damage=damage,
            attack_radius=attack_radius,
            hit_speed=hit_speed, first_hit_speed=first_hit_speed,
            target_types=target_types,
            **kwargs
        )

        self.width = width
        self.height = height

        self.size = Vector2(self.width * 16, self.height * 16)

        self.position.x = self.col * 16 
        self.position.y = self.row * 16 


    def render(self, screen, color) -> None:
        pygame.draw.rect(
            screen, color, 
            (
                self.position.x - self.size.x/2,
                self.position.y - self.size.y/2,
                self.size.x, 
                self.size.y, 
            )
        )


        # Health Bar
        health_bar_length = 30 * (self.health / self.hitpoints)

        shape = (
            self.position.x - 15,
            self.position.y - self.size.y/2 - 15,
            health_bar_length,
            4
        )

        pygame.draw.rect(screen, color, shape)
        pygame.draw.rect(screen, "black", shape, width=1)

        ### * DEBUG * ###

        # if True:
        if False:
            # Attack radius
            pygame.draw.circle(screen, "black", self.position, self.attack_radius_cells, width=1)


    def get_cell_occupancy_index(self) -> int:
        return 2


    def get_cell_occupancy(self):
        mask = np.ones((int(self.size.x), int(self.size.y))) * self.get_cell_occupancy_index()
        return mask, self.position - Vector2(self.size.x/2, self.size.y/2)
