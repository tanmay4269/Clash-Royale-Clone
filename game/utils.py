from typing import Tuple, List, Set
from rich import print

import numpy as np
import pygame

from pygame import Vector2

from game.entity import EntityType, EntityRegistry


LIGHT_BLUE = "#8BCEF7"


def smooth_arc(surface, color, rect, start_angle, stop_angle, width=1, segments=100):
    cx, cy = rect.center
    rx = rect.width / 2
    ry = rect.height / 2

    prev_point = None

    for i in range(segments + 1):
        t = i / segments
        angle = start_angle + (stop_angle - start_angle) * t

        x = cx + rx * np.cos(angle)
        y = cy + ry * np.sin(angle)

        point = (x, y)

        if prev_point:
            pygame.draw.aaline(surface, color, prev_point, point)

            # thickness (draw multiple parallel lines)
            for w in range(1, width):
                pygame.draw.aaline(surface, color,
                                   (prev_point[0], prev_point[1] + w),
                                   (point[0], point[1] + w))

        prev_point = point
