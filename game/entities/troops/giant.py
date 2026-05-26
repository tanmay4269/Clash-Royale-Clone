from game.utils import *
from game.entities.troop import Troop


@EntityRegistry.register("Giant")
class Giant(Troop):
    _image_path = "assets/giant.png"

    def __init__(self, owner, row, col, **kwargs):
        super().__init__(
            owner, row, col, 
            deploy_cost=5, deploy_delay=1.0,
            entity_type=EntityType.GROUND,
            radius=1.0, 
            speed=Troop.Speed.SLOW, 
            mass=10.0, 
            hitpoints=3968,
            damage=253,
            attack_radius=Troop.AttackRadius.MELEE_MEDIUM,
            hit_speed=1.5, first_hit_speed=0.5,
            target_types=EntityType.BUILDING,
        )
        

    def render(self, screen) -> None:
        if self.owner.side_index == 1:
            color = "red"
        else:
            color = LIGHT_BLUE

        super().render(screen, color)

        font = pygame.font.SysFont(None, 12)
        text = font.render("G", True, (0, 0, 0))  # text, antialias, color
        screen.blit(text, self.position - Vector2(3, 3))
