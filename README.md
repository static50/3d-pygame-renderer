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

The demo opens a 1600 × 800 window with a dark background and a filled red triangle. The loop is capped at 60 FPS. A cube is also created and projected, but the current draw loop only displays the triangle.

Rotation calls are present in `renderer.py`, with all angles initially set to zero. Edit those values to experiment with motion. There are no keyboard or mouse controls; close the window to exit.

## Project structure

- `renderer.py`: application entry point, demo objects, and render loop.
- `scene.py`: perspective projection using the camera's basis vectors, plus object drawing helpers.
- `camera.py`: camera position, orientation, and rotation methods.
- `prism.py`: cube edge geometry, translation, scaling, and rotations.
- `triangle.py`: triangle transforms, Bresenham line rasterization, and scanline filling.

## Current limitations

- A depth buffer is allocated, but triangle filling does not yet use it for depth testing.
- Projection does not clip geometry at the near plane or behind the camera.
- Prism rotations act around the world origin; the triangle's Y rotation uses its centroid.
- `Scene` currently starts with an empty object list even when objects are passed to its constructor. The demo projects each object directly.
- Projection dimensions and focal length are hardcoded in `scene.py`.
