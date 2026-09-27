"""Measure rendering work without the 60 FPS cap or display refresh.

Run: python benchmark_renderer.py
Uses the demo geometry at several camera distances, including near-plane cases.
"""

import os
import statistics
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from camera import Camera
from scene import Scene
from triangle import Triangle


def main():
    screen = pygame.Surface((1600, 800), depth=32)
    z_buffer = np.full((800, 1600), np.inf)
    camera = Camera([0, -500, 0])
    triangles = [
        Triangle([[0, 1, 0], [3, 1, 0], [0.5, 1, 1.73 / 2]], "red"),
        Triangle([[1, 0, 0], [2, 0, 0], [1.5, 0, 1.73 / 2]], "green"),
    ]
    for triangle in triangles:
        triangle.scale(100)
    scene = Scene(screen, triangles, camera)

    print("CPU rendering at 1600x800; excludes display refresh and FPS limiter")
    print(f"{'Camera Y':>10} {'Median ms':>12} {'95th % ms':>12} {'Visible pixels':>16}")
    for camera_y in (-500, -250, -100, -1, 99, 101):
        camera.position[1] = camera_y
        samples = []
        for frame in range(105):
            start = time.perf_counter()
            screen.fill((30, 32, 38))
            z_buffer.fill(np.inf)
            scene.project_all()
            for triangle in triangles:
                triangle.fill_triangle(screen, z_buffer)
            if frame >= 5:
                samples.append((time.perf_counter() - start) * 1000)
        visible = np.count_nonzero(np.isfinite(z_buffer))
        print(f"{camera_y:10} {statistics.median(samples):12.3f} "
              f"{np.percentile(samples, 95):12.3f} {visible:16}")


if __name__ == "__main__":
    main()
