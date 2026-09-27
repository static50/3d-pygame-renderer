import pygame
import math
import numpy as np

class Triangle:
    def __init__(self, vertices, color):
        self.vertices = np.array(vertices, dtype=float)
        self.lines = [(0, 1), (1,2), (2,0)]
        self.screen_vertices = np.array(vertices, dtype=float)
        self.color = color
        self.camera_vertices = np.array(vertices, dtype=float)
        self.screen_center = (800.0, 400.0)
        self.focal_length = 500.0
        self.near_clip = 1.0

    def move(self, x, y, z):
        for vertice in self.vertices:
            vertice[0] += x
            vertice[1] += y
            vertice[2] += z

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

        # makes the triangle rotate about the origin
        for i, vertice in enumerate(self.vertices):
            cv = vertice - centroid
            cv = rotated_basis @ cv
            self.vertices[i] = cv + centroid

    def fill_pixel(self, screen, x, y):
        screen.set_at((x, y), self.color)

    def fill_triangle(self, screen, z_buffer, color="red"):
        """Fill the near-clipped convex polygon in bounded NumPy batches.

        Sample pixel centers; include left/top and exclude right/bottom edges.
        z_buffer[y,x] stores forward camera depth, or infinity for empty pixels.
        """
        vertices = self.screen_vertices[:, :2]
        if len(vertices) < 3 or not np.isfinite(vertices).all():
            return
        clip = screen.get_clip()
        if not clip.width or not clip.height:
            return
        if (vertices[:, 0].max() <= clip.left or
                vertices[:, 0].min() >= clip.right):
            return
        y_start = max(clip.top, math.ceil(float(vertices[:, 1].min()) - 0.5))
        y_stop = min(clip.bottom, math.ceil(float(vertices[:, 1].max()) - 0.5))
        equation = self._depth_equation()
        if y_start >= y_stop or equation is None:
            return

        ys = np.arange(y_start, y_stop, dtype=float) + 0.5
        left = np.full(len(ys), np.inf)
        right = np.full(len(ys), -np.inf)
        for index in range(len(vertices)):
            x0, y0 = vertices[index - 1]
            x1, y1 = vertices[index]
            # Canonical endpoint order keeps shared-edge rounding identical.
            if y0 > y1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            if y0 == y1:
                continue
            active = (ys >= y0) & (ys < y1)
            fraction = (ys[active] - y0) / (y1 - y0)
            xs = (1.0 - fraction) * x0 + fraction * x1
            left[active] = np.minimum(left[active], xs)
            right[active] = np.maximum(right[active], xs)

        # Clamp before integer conversion. Offscreen edges never allocate
        # huge arrays or trigger long Bresenham walks.
        starts = np.ceil(np.clip(left - 0.5, clip.left, clip.right)).astype(np.intp)
        stops = np.ceil(np.clip(right - 0.5, clip.left, clip.right)).astype(np.intp)
        occupied = starts < stops
        if not occupied.any():
            return
        x_start = int(starts[occupied].min())
        x_stop = int(stops[occupied].max())
        xs = np.arange(x_start, x_stop)
        slope_x, slope_y, center_inverse = equation
        centerx, centery = self.screen_center
        x_terms = slope_x * (xs + 0.5 - centerx)
        y_terms = slope_y * (ys - centery) + center_inverse

        # Resolve named colors before locking the surface.
        resolved_color = pygame.Color(self.color)
        # Pygame uses [x,y]; transpose to match the buffer's [y,x].
        if screen.get_bitsize() == 24:
            pixels = pygame.surfarray.pixels3d(screen)
            destination = pixels.transpose(1, 0, 2)
            pixel_color = np.asarray(resolved_color[:3], dtype=np.uint8)
        else:
            # SDL can return signed packed colors for 32-bit alpha surfaces.
            mapped_color = screen.map_rgb(resolved_color) & ((1 << screen.get_bitsize()) - 1)
            pixels = pygame.surfarray.pixels2d(screen)
            destination = pixels.T
            pixel_color = np.asarray(mapped_color, dtype=pixels.dtype)
        try:
            # Bound temporary memory even for full-screen triangles.
            for row in range(0, len(ys), 32):
                end = min(row + 32, len(ys))
                valid_rows = occupied[row:end]
                if not valid_rows.any():
                    continue
                block_starts = starts[row:end]
                block_stops = stops[row:end]
                x0 = int(block_starts[valid_rows].min())
                x1 = int(block_stops[valid_rows].max())
                columns = xs[x0 - x_start:x1 - x_start]
                covered = ((columns >= block_starts[:, None]) &
                           (columns < block_stops[:, None]))
                depths = (y_terms[row:end, None] +
                          x_terms[None, x0 - x_start:x1 - x_start])
                covered &= depths > 0.0
                np.divide(1.0, depths, out=depths, where=covered)
                covered &= depths >= self.near_clip * (1.0 - 1e-12)
                buffer_view = z_buffer[y_start + row:y_start + end, x0:x1]
                covered &= depths < buffer_view
                np.copyto(buffer_view, depths, where=covered)
                color_mask = covered[:, :, None] if pixels.ndim == 3 else covered
                np.copyto(destination[y_start + row:y_start + end, x0:x1],
                          pixel_color, where=color_mask)
        finally:
            del destination
            del pixels

    def _depth_equation(self):
        """Inverse-depth slopes and its value at the screen center.

        The triangle plane is N dot P = k. Substituting the pixel ray
        P = depth * V gives depth = k / (N dot V). Since V=(1,up,right),
        inverse depth is linear on screen. This is the same intersection
        as the coupled barycentric equations, simplified once per triangle.
        """
        if not np.isfinite(self.camera_vertices).all():
            return None
        af, au, ar = map(float, self.camera_vertices[0])
        bf, bu, br = map(float, self.camera_vertices[1])
        cf, cu, cr = map(float, self.camera_vertices[2])
        e1 = (bf - af, bu - au, br - ar)
        e2 = (cf - af, cu - au, cr - ar)
        scale = max(map(abs, (*e1, *e2)))
        if scale == 0.0 or not math.isfinite(scale):
            return None
        e1f, e1u, e1r = (value / scale for value in e1)
        e2f, e2u, e2r = (value / scale for value in e2)
        nf = e1u * e2r - e1r * e2u
        nu = e1r * e2f - e1f * e2r
        nr = e1f * e2u - e1u * e2f
        plane_offset = nf * af + nu * au + nr * ar
        if plane_offset == 0.0 or not math.isfinite(plane_offset):
            return None
        equation = (nr / plane_offset / self.focal_length,
                    -nu / plane_offset / self.focal_length,
                    nf / plane_offset)
        return equation if all(map(math.isfinite, equation)) else None

    def calculate_depth(self, x, y):
        """Depth along one pixel ray; rasterization batches this calculation."""
        equation = self._depth_equation()
        if equation is None:
            return np.inf
        slope_x, slope_y, center_inverse = equation
        centerx, centery = self.screen_center
        inverse_depth = (slope_x * (x - centerx) +
                         slope_y * (y - centery) + center_inverse)
        if inverse_depth <= 0.0 or not math.isfinite(inverse_depth):
            return np.inf
        return 1.0 / inverse_depth

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
