"""Headless regression tests: python -m unittest discover -s tests -v."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from camera import Camera
from scene import Scene
from triangle import Triangle


class RendererTests(unittest.TestCase):
    size = (47, 29)
    focal = 23.5
    near = 0.2
    background = (13, 17, 23)
    footprint = np.array([[3.1, 3.2], [43.7, 3.4], [3.4, 25.8]])

    def make_scene(self, objects=(), camera=None, bits=32):
        screen = pygame.Surface(self.size, depth=bits)
        screen.fill(self.background)
        scene = Scene(screen, objects, Camera([0, 0, 0]) if camera is None else camera,
                      focal_length=self.focal, near_clip=self.near)
        depth = np.full((self.size[1], self.size[0]), np.inf)
        return scene, depth

    @staticmethod
    def from_camera(vertices, color="red", camera=None):
        camera = Camera([0, 0, 0]) if camera is None else camera
        basis = np.array([camera.forward, camera.up, camera.right])
        world = np.asarray(vertices, dtype=float) @ basis + camera.position
        return Triangle(world, color)

    def from_projection(self, depths, color="red", footprint=None):
        xy = self.footprint if footprint is None else np.asarray(footprint)
        depths = np.asarray(depths, dtype=float)
        vertices = np.column_stack((depths,
            (self.size[1] / 2 - xy[:, 1]) * depths / self.focal,
            (xy[:, 0] - self.size[0] / 2) * depths / self.focal))
        return self.from_camera(vertices, color)

    @staticmethod
    def original_depth_reference(vertices, x, y, center, focal):
        """The original coupled omega/epsilon equations, retained as an oracle."""
        a, b, c = vertices
        ray = np.array([1.0, (center[1] - y) / focal, (x - center[0]) / focal])
        constant = np.cross(a, ray)
        coefficients = np.column_stack((np.cross(b - a, ray)[1:],
                                        np.cross(c - a, ray)[1:]))
        try:
            omega, epsilon = np.linalg.solve(coefficients, -constant[1:])
        except np.linalg.LinAlgError:
            return np.inf
        point = a + omega * (b - a) + epsilon * (c - a)
        return point[0] if np.isfinite(point).all() and point[0] > 0 else np.inf

    def reference_image(self, triangle, scene):
        """Independent ray-plane intersections and 3D inside-triangle checks."""
        width, height = scene.screen.get_size()
        y, x = np.mgrid[:height, :width]
        rays = np.stack((np.ones_like(x), (height / 2 - y - 0.5) / self.focal,
                         (x + 0.5 - width / 2) / self.focal), axis=-1)
        a, b, c = triangle.camera_vertices
        edge1, edge2 = b - a, c - a
        normal = np.cross(edge1, edge2)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (normal @ a) / (rays @ normal)
            offset = rays * depth[..., None] - a
            aa, ab, bb = edge1 @ edge1, edge1 @ edge2, edge2 @ edge2
            determinant = aa * bb - ab * ab
            u = (bb * (offset @ edge1) - ab * (offset @ edge2)) / determinant
            v = (aa * (offset @ edge2) - ab * (offset @ edge1)) / determinant
        visible = ((depth >= self.near) & np.isfinite(depth) & (u >= -1e-10)
                   & (v >= -1e-10) & (u + v <= 1 + 1e-10))
        clip = scene.screen.get_clip()
        visible &= ((x >= clip.left) & (x < clip.right)
                    & (y >= clip.top) & (y < clip.bottom))
        return np.where(visible, depth, np.inf)

    def test_projection_keeps_camera_coordinates_and_configurable_parameters(self):
        camera = Camera([7, -3, 11])
        camera.rotate_z(31)
        camera.rotate_x(-17)
        original = np.array([[4, 1, -2], [7, -1, 3], [5, 2, 1]], dtype=float)
        first = self.from_camera(original, camera=camera)
        second = self.from_camera(original + [3, 0, 0], camera=camera)
        scene, _ = self.make_scene([first, second], camera)
        self.assertEqual(list(scene.obj), [first, second])
        scene.project_all()
        for triangle, expected in ((first, original), (second, original + [3, 0, 0])):
            np.testing.assert_allclose(triangle.camera_vertices, expected, atol=1e-12)
            np.testing.assert_allclose(triangle.screen_center, [23.5, 14.5])
            self.assertEqual(triangle.focal_length, self.focal)
            self.assertEqual(triangle.near_clip, self.near)
            np.testing.assert_allclose(triangle.screen_vertices[:, 0],
                                       23.5 + self.focal * expected[:, 2] / expected[:, 0])
            np.testing.assert_allclose(triangle.screen_vertices[:, 1],
                                       14.5 - self.focal * expected[:, 1] / expected[:, 0])
            np.testing.assert_allclose(triangle.screen_vertices[:, 2], expected[:, 0])

    def test_optimized_depth_matches_original_coupled_equations(self):
        random = np.random.default_rng(20260926)
        for _ in range(20):
            vertices = np.column_stack((random.uniform(2, 20, 3),
                                        random.uniform(-5, 5, (3, 2))))
            triangle = self.from_camera(vertices)
            scene, _ = self.make_scene([triangle])
            scene.project_obj(triangle)
            point = random.dirichlet([2, 2, 2]) @ vertices
            x = self.size[0] / 2 + self.focal * point[2] / point[0]
            y = self.size[1] / 2 - self.focal * point[1] / point[0]
            for sample_x, sample_y in ((x, y), (7.5, 9.5), (36.5, 18.5)):
                expected = self.original_depth_reference(vertices, sample_x, sample_y,
                                                         (23.5, 14.5), self.focal)
                np.testing.assert_allclose(triangle.calculate_depth(sample_x, sample_y),
                                           expected, rtol=1e-10, atol=1e-10)
            self.assertAlmostEqual(triangle.calculate_depth(x, y), point[0], places=10)

    def test_overlapping_triangles_are_independent_of_draw_order(self):
        for depths in (([3, 3, 3], [9, 9, 9]), ([2, 13, 7], [11.123, 3.123, 9.123])):
            with self.subTest(depths=depths):
                triangles = [self.from_projection(depths[0], "red"),
                             self.from_projection(depths[1], "blue")]
                results = []
                for order in ((0, 1), (1, 0)):
                    scene, z_buffer = self.make_scene(triangles)
                    scene.project_all()
                    references = [self.reference_image(t, scene) for t in triangles]
                    expected = np.minimum(*references)
                    for index in order:
                        triangles[index].fill_triangle(scene.screen, z_buffer)
                    np.testing.assert_allclose(z_buffer, expected, rtol=1e-10, atol=1e-10)
                    colors = pygame.surfarray.array3d(scene.screen).transpose(1, 0, 2)
                    red, blue = references[0] < references[1], references[1] < references[0]
                    self.assertTrue(np.all(colors[red] == (255, 0, 0)))
                    self.assertTrue(np.all(colors[blue] == (0, 0, 255)))
                    self.assertTrue(np.all(colors[~np.isfinite(expected)] == self.background))
                    self.assertGreater(np.count_nonzero(np.isfinite(expected)), 100)
                    if depths[0][0] == 2:
                        self.assertTrue(red.any() and blue.any())
                    results.append((colors, z_buffer))
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_allclose(results[0][1], results[1][1])

    def test_clipped_polygon_sizes_and_visible_depths(self):
        cases = (
            ([[1, 0.3, -0.3], [1, 0.3, 0.3], [1, -0.3, 0]], 3),
            ([[1, 0.8, -0.8], [1, 0.8, 0.8], [-0.6, -0.8, 0]], 4),
            ([[1, 0.53, 0.017], [-0.6, -0.83, -0.81], [-0.6, -0.83, 0.79]], 3),
            ([[1, 0.5, 0], [0, -0.8, -0.8], [0, -0.8, 0.8]], 3),
            ([[-1, 0.3, -0.3], [-1, 0.3, 0.3], [-1, -0.3, 0]], 0),
        )
        for vertices, polygon_size in cases:
            with self.subTest(vertices=vertices):
                triangle = self.from_camera(vertices)
                scene, z_buffer = self.make_scene([triangle])
                scene.project_all()
                self.assertEqual(triangle.screen_vertices.shape, (polygon_size, 3))
                np.testing.assert_allclose(triangle.camera_vertices, vertices)
                self.assertTrue(np.isfinite(triangle.screen_vertices).all())
                self.assertTrue(np.all(triangle.screen_vertices[:, 2] >= self.near))
                expected = self.reference_image(triangle, scene)
                triangle.fill_triangle(scene.screen, z_buffer)
                np.testing.assert_allclose(z_buffer, expected, rtol=1e-9, atol=1e-9)
                self.assertEqual(np.isfinite(z_buffer).any(), polygon_size > 0)

    def test_moving_behind_camera_discards_previous_projection(self):
        triangle = self.from_projection([3, 3, 3])
        scene, z_buffer = self.make_scene([triangle])
        scene.project_all()
        triangle.fill_triangle(scene.screen, z_buffer)
        self.assertTrue(np.isfinite(z_buffer).any())
        triangle.move(0, -5, 0)
        scene.project_all()
        self.assertEqual(triangle.screen_vertices.shape, (0, 3))
        scene.screen.fill(self.background)
        z_buffer.fill(np.inf)
        triangle.fill_triangle(scene.screen, z_buffer)
        self.assertTrue(np.isinf(z_buffer).all())
        self.assertTrue(np.all(pygame.surfarray.array3d(scene.screen) == self.background))

    def test_shared_diagonal_has_no_holes_for_either_winding(self):
        corners = np.array([[8, 5], [28, 5], [28, 25], [8, 25]])
        for reverse in (False, True):
            scene, z_buffer = self.make_scene()
            for indices in ([0, 1, 2], [0, 2, 3]):
                if reverse:
                    indices = indices[::-1]
                triangle = self.from_projection([4, 4, 4], footprint=corners[indices])
                scene.project_obj(triangle)
                triangle.fill_triangle(scene.screen, z_buffer)
            expected = np.zeros(z_buffer.shape, dtype=bool)
            expected[5:25, 8:28] = True
            np.testing.assert_array_equal(np.isfinite(z_buffer), expected)
            np.testing.assert_allclose(z_buffer[expected], 4)
            colors = pygame.surfarray.array3d(scene.screen).transpose(1, 0, 2)
            self.assertTrue(np.all(colors[expected] == (255, 0, 0)))

    def test_giant_near_edges_are_bounded_and_respect_clip(self):
        for footprint, visible in (
            ([[-1e9, -1e9], [3e9, -1e9], [-1e9, 3e9]], True),
            ([[1e6, 1e6], [1e6 + 100, 1e6], [1e6, 1e6 + 100]], False),
        ):
            scene, z_buffer = self.make_scene()
            clip = pygame.Rect(5, 6, 13, 10)
            scene.screen.set_clip(clip)
            triangle = self.from_projection([0.200001] * 3, footprint=footprint)
            scene.project_obj(triangle)
            with patch.object(Triangle, "rasterize_line", create=True,
                              side_effect=AssertionError("Do not enumerate offscreen edges")):
                triangle.fill_triangle(scene.screen, z_buffer)
            expected = np.zeros(z_buffer.shape, dtype=bool)
            if visible:
                expected[clip.top:clip.bottom, clip.left:clip.right] = True
            np.testing.assert_array_equal(np.isfinite(z_buffer), expected)
            np.testing.assert_allclose(z_buffer[expected], 0.200001)
            colors = pygame.surfarray.array3d(scene.screen).transpose(1, 0, 2)
            self.assertTrue(np.all(colors[expected] == (255, 0, 0)))
            self.assertTrue(np.all(colors[~expected] == self.background))

    def test_shared_edge_rounding_gives_boundary_pixel_exactly_one_owner(self):
        a = (-9.982251377974654, -3.8254747953388337)
        b = (15.074421339159196, 11.266992325502697)
        for reverse in (False, True):
            owners = 0
            for footprint in ([a, b, (0, 10)], [b, a, (15, 0)]):
                if reverse:
                    footprint = footprint[::-1]
                triangle = self.from_projection([4, 4, 4], footprint=footprint)
                scene, z_buffer = self.make_scene([triangle])
                scene.project_all()
                # Preserve the exact edge ulps: reversed interpolation otherwise
                # puts x just above/below 5.5 at y=5.5. Reprojection rounds them.
                triangle.screen_vertices[:, :2] = footprint
                triangle.fill_triangle(scene.screen, z_buffer)
                owners += int(np.isfinite(z_buffer[5, 5]))
            self.assertEqual(owners, 1)

    def test_degenerate_or_behind_geometry_does_not_produce_depth(self):
        for vertices in (
            [[2, 0, 0], [2, 0, 0], [2, 0, 0]],
            [[2, 0, 0], [2, 1, 1], [2, 2, 2]],
            [[2, 0, -1], [5, 0, 1], [3, 0, 3]],
            [[-2, -1, -1], [-2, 1, -1], [-2, 0, 1]],
        ):
            triangle = self.from_camera(vertices)
            scene, z_buffer = self.make_scene([triangle])
            scene.project_all()
            triangle.fill_triangle(scene.screen, z_buffer)
            self.assertTrue(np.isinf(z_buffer).all())
            self.assertEqual(triangle.calculate_depth(23.5, 14.5), np.inf)

    def test_camera_accepts_fractional_movement_and_projection_updates(self):
        camera = Camera([0, 0, 0])
        triangle = self.from_camera([[4, 1, -2], [7, -1, 3], [5, 2, 1]])
        scene, _ = self.make_scene([triangle], camera)
        scene.project_all()
        before = triangle.camera_vertices.copy()
        camera.position += np.array([0.25, 0.5, -0.125])
        scene.project_all()
        np.testing.assert_allclose(camera.position, [0.25, 0.5, -0.125])
        np.testing.assert_allclose(triangle.camera_vertices, before - [0.5, -0.125, 0.25])

    def test_24_and_32_bit_surfaces_accept_colors_and_release_pixel_views(self):
        for bits in (24, 32):
            for color in ("red", (17, 193, 88)):
                with self.subTest(bits=bits, color=color):
                    triangle = self.from_projection([3, 3, 3], color)
                    scene, z_buffer = self.make_scene([triangle], bits=bits)
                    scene.project_all()
                    triangle.fill_triangle(scene.screen, z_buffer)
                    self.assertFalse(scene.screen.get_locked())
                    expected_color = pygame.Color(color)[:3]
                    self.assertEqual(scene.screen.get_at((10, 8))[:3], expected_color)
                    self.assertAlmostEqual(z_buffer[8, 10], 3)
                    destination = pygame.Surface(self.size)
                    destination.blit(scene.screen, (0, 0))
                    self.assertEqual(destination.get_at((10, 8))[:3], expected_color)

    def test_alpha_surface_preserves_packed_color_without_signed_overflow(self):
        for color in ("red", (17, 193, 88, 77)):
            with self.subTest(color=color):
                triangle = self.from_projection([3, 3, 3], color)
                scene, z_buffer = self.make_scene([triangle])
                scene.screen = pygame.Surface(self.size, pygame.SRCALPHA, 32)
                scene.project_all()
                triangle.fill_triangle(scene.screen, z_buffer)
                self.assertEqual(scene.screen.get_at((10, 8)), pygame.Color(color))
                self.assertAlmostEqual(z_buffer[8, 10], 3)
                self.assertFalse(scene.screen.get_locked())

    def test_surface_pixel_views_are_released_if_rendering_fails(self):
        triangle = self.from_projection([3, 3, 3], (255, 0, 0))
        for bits in (24, 32):
            with self.subTest(bits=bits):
                scene, z_buffer = self.make_scene([triangle], bits=bits)
                scene.project_all()
                with patch("triangle.np.copyto", side_effect=RuntimeError("forced copy failure")):
                    with self.assertRaisesRegex(RuntimeError, "forced copy failure"):
                        triangle.fill_triangle(scene.screen, z_buffer)
                self.assertFalse(scene.screen.get_locked())


if __name__ == "__main__":
    unittest.main()
