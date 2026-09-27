import pygame
import numpy as np
import prism
import triangle
import camera
import scene

'''
TO DO:
- make the camera moveable with WSAD and mouse
- add far plane clipping
'''
FPS = 60
CUBE_SCALE = 100
TRIANGLE_SCALE = 100
CAMERA_POSITION = np.array([0, -500, 0])
BACKGROUND_COLOR = (30, 32, 38)
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800
FOCAL_LENGTH = 500
NEAR_PLANE = 1.0

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
        [0, 1, 0],
        [3, 1, 0],
        [0.5, 1, 1.73/2],
    ]

    vertices_triangle2 = [  # (x, z, y)
        [0 + 1, 0, 0],
        [1 + 1, 0, 0],
        [0.5 + 1, 0, 1.73 / 2],
    ]

    # create our cube with our vertices
    cube = prism.Prism(vertices)
    cube.scale(CUBE_SCALE)

    t1 = triangle.Triangle(vertices_triangle, "red")
    t1.scale(TRIANGLE_SCALE)

    t2 = triangle.Triangle(vertices_triangle2, "green")
    t2.scale(TRIANGLE_SCALE)

    objects = scene.Scene(screen, [t1, t2], cam,
                          focal_length=FOCAL_LENGTH, near_clip=NEAR_PLANE)


    while running:
        dt = clock.tick(FPS) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            cam.handle_event(event)

        if not running:
            break

        cam.position[1] += dt * 100
        cam.position[0] -= dt * 15
        cam.rotate_z(-0.3)
        screen.fill(BACKGROUND_COLOR)
        z_buffer.fill(np.inf)

        objects.project_all()

        t1.fill_triangle(screen, z_buffer)
        t2.fill_triangle(screen, z_buffer)

        pygame.display.flip()


    pygame.quit()










if __name__ == "__main__":
    main()
