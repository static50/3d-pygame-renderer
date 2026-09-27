import pygame
import numpy as np
import math

class Camera:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        # camera basis
        self.forward = [0, 1, 0]
        self.up = [0, 0, 1]
        self.right = [1, 0, 0]

        # mouse location
        self.mousex = 0
        self.mousey = 0

    def handle_event(self, event):
        if event == pygame.MOUSEMOTION:
            print(pygame.mouse.get_pos())
        print(pygame.mouse.get_pos())



    def rotate_z(self, z_angle):
        rotation_matrix_z = np.array([
            [math.cos(math.radians(z_angle)), -math.sin(math.radians(z_angle)), 0],
            [math.sin(math.radians(z_angle)), math.cos(math.radians(z_angle)), 0],
            [0, 0, 1]
        ])

        self.forward = np.dot(rotation_matrix_z, self.forward)
        self.right = np.dot(rotation_matrix_z, self.right)
        self.up = np.dot(rotation_matrix_z, self.up)

    def rotate_x(self, x_angle):
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, math.cos(math.radians(x_angle)), -math.sin(math.radians(x_angle))],
            [0, math.sin(math.radians(x_angle)), math.cos(math.radians(x_angle))]
        ])

        self.forward = np.dot(rotation_matrix_x, self.forward)
        self.right = np.dot(rotation_matrix_x, self.right)
        self.up = np.dot(rotation_matrix_x, self.up)

    def rotate_y(self, y_angle):
        rotation_matrix_y = np.array([
            [math.cos(math.radians(y_angle)), 0, math.sin(math.radians(y_angle))],
            [0, 1, 0],
            [-math.sin(math.radians(y_angle)), 0, math.cos(math.radians(y_angle))]
        ])

        self.forward = np.dot(rotation_matrix_y, self.forward)
        self.right = np.dot(rotation_matrix_y, self.right)
        self.up = np.dot(rotation_matrix_y, self.up)
