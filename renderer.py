from codecs import xmlcharrefreplace_errors

import pygame
import numpy as np
import prism
import triangle
import math
import camera
import scene

'''
TO DO: 
- make homogenous coordinates (who the fuck came up with ts?)
- make view, projection, and model matricies (or not, tbh idc)
- make a z-buffer that does z buffer things


'''
FPS = 60
CUBE_SCALE = 100
TRIANGLE_SCALE = 100
CAMERA_POSITION = np.array([0, -500, 0])
BACKGROUND_COLOR = (30, 32, 38)
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800

centerx = int(WINDOW_WIDTH / 2)
centery = int(WINDOW_HEIGHT / 2)
FOCAL_LENGTH = 500

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    running = True
    clock = pygame.time.Clock()

    z_buffer = np.full((WINDOW_HEIGHT, WINDOW_WIDTH), np.inf)
    cam = camera.Camera(CAMERA_POSITION)

    # define vertices
    vertices = [
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1]
    ]

    vertices_triangle = [ # (x, z, y)
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0, 1.73/2],
    ]

    # create our cube with our vertices
    cube = prism.Prism(vertices)
    cube.scale(CUBE_SCALE)

    t1 = triangle.Triangle(vertices_triangle)
    t1.scale(TRIANGLE_SCALE)

    objects = scene.Scene(screen, [cube, t1], cam)

    while running:
        dt = clock.tick(FPS) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BACKGROUND_COLOR)
        z_buffer.fill(np.inf)

        # apply some transformations

        # -----------------------------------------------------
        # BUG: THAT THE CUBE IS ROTATING ABOUT THE WORLD ORIGIN.
        # -----------------------------------------------------
        cube.rotate_x(0) # btw this will rotate every frame
        cube.rotate_y(0)
        cube.rotate_z(0)

        t1.rotate_y(0)

        # apply perspective & camera rotation
        objects.cam.rotate_z(0)
        objects.project_obj(t1)
        objects.project_obj(cube)


        t1.fill_triangle(screen, z_buffer)
        draw_obj(screen, t1)

        pygame.display.flip()


    pygame.quit()

def draw_obj(screen, primitive):
    for start, end in primitive.lines:
        # 3d to 2d projection
        start_pos = primitive.screen_vertices[start][:2]
        end_pos = primitive.screen_vertices[end][:2]

        # draw the primitive
        pygame.draw.line(screen, "red", start_pos, end_pos, 5)











if __name__ == "__main__":
    main()
