import numpy as np
import pygame

from triangle import Triangle


NEAR_PLANE_BOUNDARY = 1.0


class Scene:
    def __init__(self, screen, obj, cam, focal_length=500.0,
                 near_clip=NEAR_PLANE_BOUNDARY):
        if not np.isfinite(focal_length) or focal_length <= 0:
            raise ValueError("focal_length must be positive and finite")
        if not np.isfinite(near_clip) or near_clip <= 0:
            raise ValueError("near_clip must be positive and finite")
        self.obj = list(obj)
        self.cam = cam
        self.screen = screen
        self.focal_length = float(focal_length)
        self.near_clip = float(near_clip)

    def draw_obj(self, primitive):
        """Draw a clipped debug wireframe overlay without depth testing."""
        clip = self.screen.get_clip()
        if clip.width == 0 or clip.height == 0:
            return

        for start, end in primitive.lines:
            first = primitive.camera_vertices[start]
            second = primitive.camera_vertices[end]
            if not np.isfinite([first, second]).all():
                continue

            first_inside = first[0] >= self.near_clip
            second_inside = second[0] >= self.near_clip
            if not first_inside and not second_inside:
                continue
            if first_inside != second_inside:
                intersection = self._near_intersection(first, second)
                if first_inside:
                    second = intersection
                else:
                    first = intersection

            projected = self._project_vertices(np.array([first, second]))
            segment = self._clip_screen_line(projected[0], projected[1], clip)
            if segment is not None:
                pygame.draw.line(self.screen, primitive.color, *segment, 5)

    def draw_all(self):
        """Draw debug wireframe overlays; these do not use the z-buffer."""
        for primitive in self.obj:
            self.draw_obj(primitive)

    def project_all(self):
        if not self.obj:
            return
        # Build the camera transform once for every object in this frame.
        basis = np.array([self.cam.forward, self.cam.up, self.cam.right]).T
        world_to_camera = np.linalg.solve(basis, np.eye(3))
        for primitive in self.obj:
            camera_vertices = (
                (primitive.vertices - self.cam.position) @ world_to_camera.T
            )
            self._set_projected_vertices(primitive, camera_vertices)

    def add_obj(self, obj):
        self.obj.append(obj)

    def project_obj(self, primitive):
        # One solve transforms all vertices, instead of solving once per vertex.
        basis = np.array([self.cam.forward, self.cam.up, self.cam.right]).T
        camera_vertices = np.linalg.solve(
            basis, (primitive.vertices - self.cam.position).T
        ).T
        self._set_projected_vertices(primitive, camera_vertices)

    def _set_projected_vertices(self, primitive, camera_vertices):
        primitive.camera_vertices = camera_vertices
        width, height = self.screen.get_size()
        primitive.screen_center = (width / 2, height / 2)
        primitive.focal_length = self.focal_length
        primitive.near_clip = self.near_clip

        if isinstance(primitive, Triangle):
            # Preserve the original three camera vertices for the depth plane;
            # the visible polygon may have zero, three, or four vertices.
            visible = self._clip_near_polygon(camera_vertices)
            primitive.screen_vertices = self._project_vertices(visible)
        else:
            # Indexed wireframes must keep the original vertex numbering.
            projected = np.full_like(camera_vertices, np.nan)
            visible = (
                np.isfinite(camera_vertices).all(axis=1)
                & (camera_vertices[:, 0] >= self.near_clip)
            )
            projected[visible] = self._project_vertices(camera_vertices[visible])
            primitive.screen_vertices = projected

    def _project_vertices(self, camera_vertices):
        if len(camera_vertices) == 0:
            return np.empty((0, 3), dtype=float)
        width, height = self.screen.get_size()
        depth = camera_vertices[:, 0]
        return np.column_stack((
            width / 2 + self.focal_length * camera_vertices[:, 2] / depth,
            height / 2 - self.focal_length * camera_vertices[:, 1] / depth,
            depth,
        ))

    def _near_intersection(self, first, second):
        # Shared triangle edges must yield the same cut vertex in either order.
        if first[0] > second[0]:
            first, second = second, first
        if first[0] == self.near_clip:
            return first.copy()
        if second[0] == self.near_clip:
            return second.copy()
        fraction = (self.near_clip - first[0]) / (second[0] - first[0])
        intersection = (1 - fraction) * first + fraction * second
        intersection[0] = self.near_clip
        return intersection

    def _clip_near_polygon(self, vertices):
        if len(vertices) == 0 or not np.isfinite(vertices).all():
            return np.empty((0, 3), dtype=float)
        if np.all(vertices[:, 0] >= self.near_clip):
            return vertices.copy()
        if np.all(vertices[:, 0] < self.near_clip):
            return np.empty((0, 3), dtype=float)

        clipped = []

        def append_unique(vertex):
            if not clipped or not np.array_equal(clipped[-1], vertex):
                clipped.append(vertex)

        previous = vertices[-1]
        previous_inside = previous[0] >= self.near_clip
        for current in vertices:
            current_inside = current[0] >= self.near_clip
            if current_inside != previous_inside:
                append_unique(self._near_intersection(previous, current))
            if current_inside:
                append_unique(current)
            previous = current
            previous_inside = current_inside

        if len(clipped) > 1 and np.array_equal(clipped[0], clipped[-1]):
            clipped.pop()
        if len(clipped) < 3:
            return np.empty((0, 3), dtype=float)
        return np.array(clipped, dtype=float)

    @staticmethod
    def _clip_screen_line(first, second, clip):
        """Clip float endpoints before passing bounded coordinates to pygame."""
        x, y = first[:2]
        dx, dy = second[:2] - first[:2]
        if not np.isfinite([x, y, dx, dy]).all():
            return None

        lower, upper = 0.0, 1.0
        for direction, distance in (
            (-dx, x - clip.left),
            (dx, clip.right - 1 - x),
            (-dy, y - clip.top),
            (dy, clip.bottom - 1 - y),
        ):
            if direction == 0:
                if distance < 0:
                    return None
                continue
            fraction = distance / direction
            if direction < 0:
                lower = max(lower, fraction)
            else:
                upper = min(upper, fraction)
            if lower > upper:
                return None
        return ((x + lower * dx, y + lower * dy),
                (x + upper * dx, y + upper * dy))
