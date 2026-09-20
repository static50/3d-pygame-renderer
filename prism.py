import pygame
import math
import numpy as np

class Prism:
    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=float) # these should always be in world coordinates I'm not too worried about object local coordinates rn
        self.lines = [ (0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 7), (7,3), (7,6), (6, 5), (5, 4), (6,2), (5,1) ]
        self.screen_vertices = np.array(vertices, dtype=float)

        # TO DO:
        # make an object local coordinate system with its own basis. e.g (1,1,1) is always the corner of a cube no matter if it rotates


    def move(self, x, y, z):
        for vertice in self.vertices:
            vertice[0] += x
            vertice[1] += y
            vertice[2] += z

    def rotate_z (self, z_angle):
        rotation_matrix_z = np.array([
            [math.cos(math.radians(z_angle)), -math.sin(math.radians(z_angle)), 0],
            [math.sin(math.radians(z_angle)), math.cos(math.radians(z_angle)), 0],
            [0, 0, 1]
        ])

        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])

        i = np.dot(rotation_matrix_z, i)
        j = np.dot(rotation_matrix_z, j)
        k = np.dot(rotation_matrix_z, k)

        self.vertices = np.array([i*x + j*y + k*z for x,y,z in self.vertices])


    def rotate_x (self, x_angle):
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, math.cos(math.radians(x_angle)), -math.sin(math.radians(x_angle))],
            [0, math.sin(math.radians(x_angle)), math.cos(math.radians(x_angle))]
        ])

        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])

        i = np.dot(rotation_matrix_x, i)
        j = np.dot(rotation_matrix_x, j)
        k = np.dot(rotation_matrix_x, k)

        self.vertices = np.array([i * x + j * y + k * z for x, y, z in self.vertices])

    def rotate_y (self, y_angle):
        rotation_matrix_y = np.array([
            [math.cos(math.radians(y_angle)), 0, math.sin(math.radians(y_angle))],
            [0, 1, 0],
            [-math.sin(math.radians(y_angle)), 0, math.cos(math.radians(y_angle))]
        ])

        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])
        
        i = np.dot(rotation_matrix_y, i)
        j = np.dot(rotation_matrix_y, j)
        k = np.dot(rotation_matrix_y, k)

        self.vertices = np.array([i * x + j * y + k * z for x, y, z in self.vertices])

    def scale(self, scale_factor):
        self.vertices *= scale_factor
