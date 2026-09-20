import pygame
import math
import numpy as np

class Triangle:
    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=float)
        self.lines = [(0, 1), (1,2), (2,0)]
        self.screen_vertices = np.array(vertices, dtype=float)


    def scale(self, scale_factor):
        self.vertices *= scale_factor

    def rotate_y(self, y_angle):
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

        rotated_basis = np.array([i, j, k]).T
        centroid = np.mean(self.vertices, axis=0)

        for i, vertice in enumerate(self.vertices):
            cv = vertice - centroid
            cv = rotated_basis @ cv
            self.vertices[i] = cv + centroid

    def fill_pixel(self, screen, x, y, color):
        screen.set_at((x, y), color)

    def fill_triangle(self, screen, z_buffer, color="red"):
        """Fill the projected triangle one horizontal row at a time."""
        if not np.isfinite(self.screen_vertices[:, :2]).all():
            return

        clip = screen.get_clip()
        row_bounds = {}

        # Find the leftmost and rightmost edge pixel on each screen row.
        for start, end in self.lines:
            x1 = self.screen_vertices[start][0]
            y1 = self.screen_vertices[start][1]
            x2 = self.screen_vertices[end][0]
            y2 = self.screen_vertices[end][1]
            pixels = self.rasterize_line(x1, y1, x2, y2)

            for x, y in pixels:
                if not clip.top <= y < clip.bottom:
                    continue

                left, right = row_bounds.get(y, (x, x))
                row_bounds[y] = (min(left, x), max(right, x))

        # Include both endpoints, and only draw inside the screen's clip area.
        for y, (left, right) in row_bounds.items():
            left = max(left, clip.left)
            right = min(right, clip.right - 1)
            for x in range(left, right + 1):
                self.fill_pixel(screen, x, y, color)
    @staticmethod
    def rasterize_line(x1, y1, x2, y2):
        """
        Generates pixel coordinates tracking a straight line from (x1, y1) to (x2, y2)
        using Bresenham's Line Algorithm.
        """
        # Projected coordinates are floats; pixel steps need integer endpoints.
        x1, y1, x2, y2 = (int(round(value)) for value in (x1, y1, x2, y2))
        pixels = []

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        step_x = 1 if x1 < x2 else -1
        step_y = 1 if y1 < y2 else -1

        error = dx - dy

        x, y = x1, y1

        while True:
            pixels.append((x, y))

            if x == x2 and y == y2:
                break

            error_2 = 2 * error

            if error_2 > -dy:
                error -= dy
                x += step_x

            if error_2 < dx:
                error += dx
                y += step_y

        return pixels



