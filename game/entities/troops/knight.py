from game.utils import *
from game.entities.troop import Troop


@EntityRegistry.register("Knight")
class Knight(Troop):
    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner, row, col, 
            deploy_cost=3, deploy_delay=1.0,
            entity_type=EntityType.GROUND,
            radius=0.5, 
            speed=Troop.Speed.MEDIUM,
            mass=1.0, 
            hitpoints=1766,
            damage=202,
            attack_radius=Troop.AttackRadius.MELEE_MEDIUM,
            hit_speed=1.2, first_hit_speed=0.5,
            target_types=EntityType.get_all(),
        )
        
        # self.sprite = pygame.image.load("assets/knight.png")
        # self.sprite = pygame.transform.scale(self.sprite, 2 * Vector2(self.size, self.size))


    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        # screen.blit(self.sprite, self.position - Vector2(self.size, self.size))

        font = pygame.font.SysFont(None, 12)
        text = font.render("K", True, (0, 0, 0))  # text, antialias, color
        screen.blit(text, self.position - Vector2(3, 3))
