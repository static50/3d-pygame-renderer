import numpy as np
import math
import pygame

class Scene:
    def __init__(self, screen, obj, cam):
        self.obj = []
        self.cam = cam
        self.screen = screen

    def draw_obj(self, primitive):
        for start, end in primitive.lines:
            # 3d to 2d projection
            start_pos = primitive.screen_vertices[start][:2]
            end_pos = primitive.screen_vertices[end][:2]

            # draw the primitive
            pygame.draw.line(self.screen, "red", start_pos, end_pos, 5)

    def draw_all(self):
        for object in self.obj:
            self.draw_obj(object)

    def project_all(self):
        for object in self.obj:
            self.project_obj(object)

    def add_obj(self, obj):
        self.obj.append(obj)

    def project_obj(self, object):
        FOCAL_LENGTH = 500
        WINDOW_WIDTH = 1600
        WINDOW_HEIGHT = 800
        
        centerx = int(WINDOW_WIDTH / 2)
        centery = int(WINDOW_HEIGHT / 2)
        
        for i, vertex in enumerate(object.vertices):
            # evil change of base matrix
            CV = vertex - self.cam.position
            B = np.array([self.cam.forward, self.cam.up, self.cam.right]).T  # change of basis matrix

            # Solve for the self.camera to vertex vector relative to the self.camera
            cf, cu, cr = np.linalg.solve(B, CV)

            x = cr
            depth = cf
            z = cu

            # -----------------------------------------------------
            # BUG: NO NEAR PLANE CLIPPING AS THE DEPTH APPROACHES ZERO
            # -----------------------------------------------------
            # convert to screen coordinates
            screen_x = centerx + FOCAL_LENGTH * (x / depth)
            screen_z = centery - FOCAL_LENGTH * (z / depth)
            object.screen_vertices[i] = np.array([screen_x, screen_z, depth])
            
