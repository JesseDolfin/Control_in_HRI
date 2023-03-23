import numpy as np

class GodObject:
    def __init__(self, initial_pos):
        self.pos = initial_pos
        self.prev_pos = initial_pos

    def set_pos(self, pos):
        self.prev_pos = self.pos
        self.pos = pos

    def move_along_surface(self, surface):
        surface_normal = surface.normal
        surface_point = surface.point
        distance_to_surface = np.dot(surface_point - self.pos, surface_normal)
        self.set_pos(self.pos + (distance_to_surface * surface_normal))

    def calculate_force(self, stiffness, damping, target_pos):
        force_direction = target_pos - self.pos
        force = stiffness * force_direction - damping * (self.pos - self.prev_pos)
        return force
