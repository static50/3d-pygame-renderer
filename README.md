# 3D Pygame Renderer

A work-in-progress software renderer written in Python with Pygame and NumPy. It experiments with 3D transformations, camera-relative perspective projection, and pixel-based triangle filling.

## Setup and run

Install Python 3.12 or newer (tested with Python 3.14), then run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python renderer.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. A desktop display is required to see the Pygame window.

## Current demo

The demo opens a 1600 × 800 window with two filled triangles and a depth buffer. The loop is capped at 60 FPS. A cube is also created for experimentation, but is not projected or drawn by the current loop.

Camera movement and input handling live in `camera.py`. Close the window to exit. `FOCAL_LENGTH` and `NEAR_PLANE` in `renderer.py` configure projection and clipping.

## Rendering and performance

Camera coordinates are `(forward, up, right)`. Geometry is clipped against the near plane before perspective projection. A clipped triangle can become a four-vertex polygon, which the rasterizer fills directly.

The triangle plane determines a depth equation once per triangle. Rasterization evaluates it in NumPy batches of screen pixels and writes colors through a Pygame surface array. The depth buffer still contains ordinary forward depth, with `np.inf` for empty pixels. `Triangle.calculate_depth(x, y)` remains available for inspecting an individual ray; the fill loop does not call it per pixel.

Coverage uses pixel centers and a shared-edge rule so adjacent triangles do not leave cracks. Work is bounded to the screen's clip rectangle, including when projected edges extend far offscreen.

Run the regression tests and uncapped CPU-rendering benchmark with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe benchmark_renderer.py
```

The benchmark includes screen/depth clearing, projection, and filling. It excludes display refresh and the deliberate wait imposed by the 60 FPS cap.

## Project structure

- `renderer.py`: application entry point, demo objects, and render loop.
- `scene.py`: perspective projection using the camera's basis vectors, plus object drawing helpers.
- `camera.py`: camera position, orientation, and rotation methods.
- `prism.py`: cube edge geometry, translation, scaling, and rotations.
- `triangle.py`: triangle transforms, depth calculation, and batched polygon filling.
- `benchmark_renderer.py`: repeatable rendering measurements at several camera distances.
- `tests/test_renderer.py`: depth, clipping, coverage, and surface-write regression tests.

## Current limitations

- There is no far-plane clipping or texture mapping yet.
- `Scene.draw_obj` and `draw_all` are optional debug wireframe overlays without depth testing. The filled demo does not call them.
- Prism rotations act around the world origin; the triangle's Y rotation uses its centroid.
