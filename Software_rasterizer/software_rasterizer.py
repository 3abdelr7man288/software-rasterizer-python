#Allows you to use modern type hinting without errors in older Python versions.
from __future__ import annotations
#Standard Python libraries for structured data, file paths, math operations (like sine/cosine), and operating system commands.
from dataclasses import dataclass
from pathlib import Path
import math
import os
#The core math library used here. It handles arrays and matrix multiplications
import numpy as np
#used at the very end to create and save the final 2D image array into a .png file.
from PIL import Image

#The resolution of the final rendered image.
WIDTH = 960
HEIGHT = 720
# RGB background color for pixels that are never covered by any triangle.
BACKGROUND = np.array([245.0, 248.0, 252.0], dtype=float)
# A stronger base light keeps faces readable even when they do not directly face a lamp.
AMBIENT_LIGHT = 0.35


@dataclass
class Light:
    # Represents a light bulb in the scene. It has a position in 3D space, an RGB color, and an overall intensity multiplier.
    position: np.ndarray
    color: np.ndarray
    intensity: float


@dataclass
class Camera:
    # Camera parameters for building the view and projection matrices.
    eye: np.ndarray
    target: np.ndarray #The 3D point the camera is looking at.
    up: np.ndarray#Y-axis: [0, 1, 0]
    fov_degrees: float
    near: float
    far: float


@dataclass
class Mesh:
    # Geometry plus the transform that places the object in world space.
    name: str
    vertices: np.ndarray #The raw 3D points.
    triangles: list[tuple[int, int, int]]#How those points connect to form faces.
    colors: np.ndarray
    model_matrix: np.ndarray #A $4 \times 4$ transformation matrix that dictates where this object is placed in the world, its rotation, and its scale.


def normalize(vector: np.ndarray) -> np.ndarray:
    # Return a unit-length version of the vector so lighting and directions behave correctly.
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.copy()
    return vector / norm


def translation(tx: float, ty: float, tz: float) -> np.ndarray:
    # 4x4 homogeneous translation matrix.
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    # 4x4 homogeneous scaling matrix.
    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = sx
    matrix[1, 1] = sy
    matrix[2, 2] = sz
    return matrix


def rotation_x(angle_radians: float) -> np.ndarray:
    # Rotate around the x-axis.
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def rotation_y(angle_radians: float) -> np.ndarray:
    # Rotate around the y-axis.
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def rotation_z(angle_radians: float) -> np.ndarray:
    # Rotate around the z-axis.
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def look_at(camera: Camera) -> np.ndarray:
    # Build the view matrix that moves world coordinates into camera space.
    gaze = normalize(camera.target - camera.eye)
    w = normalize(-gaze)
    u = normalize(np.cross(camera.up, w))
    v = np.cross(w, u)

    rotation = np.array(
        [
            [u[0], u[1], u[2], 0],
            [v[0], v[1], v[2], 0],
            [w[0], w[1], w[2], 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    translate = translation(-camera.eye[0], -camera.eye[1], -camera.eye[2])
    return rotation @ translate


def perspective(fov_degrees: float, aspect_ratio: float, near: float, far: float) -> np.ndarray:
    # Create a standard perspective projection matrix.
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0] = f / aspect_ratio
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def viewport(width: int, height: int, zmin: float = 0.0, zmax: float = 1.0) -> np.ndarray:
    # Map normalized device coordinates from [-1, 1] into pixel coordinates.
    return np.array(
        [
            [width / 2.0, 0, 0, (width - 1) / 2.0],
            [0, height / 2.0, 0, (height - 1) / 2.0],
            [0, 0, (zmax - zmin) / 2.0, (zmax + zmin) / 2.0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def to_homogeneous(vertices: np.ndarray) -> np.ndarray:
    # Add a w = 1 column so 3D points can be transformed by 4x4 matrices.
    ones = np.ones((vertices.shape[0], 1), dtype=float)
    return np.hstack([vertices, ones])


def transform_points(matrix: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    # Apply one transformation matrix to every vertex in the array.
    transformed = (matrix @ to_homogeneous(vertices).T).T
    return transformed


def perspective_divide(vertices_clip: np.ndarray) -> np.ndarray:
    # Convert clip-space coordinates into normalized device coordinates by dividing by w.
    ndc = vertices_clip[:, :3] / vertices_clip[:, 3:4]
    return np.hstack([ndc, np.ones((vertices_clip.shape[0], 1), dtype=float)])


def compute_vertex_normals(vertices: np.ndarray, triangles: list[tuple[int, int, int]]) -> np.ndarray:
    # Average adjacent face normals to get a smooth lighting normal at each vertex.
    normals = np.zeros_like(vertices, dtype=float)
    for i0, i1, i2 in triangles:
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        face_normal = np.cross(p1 - p0, p2 - p0)
        face_normal = normalize(face_normal)
        normals[i0] += face_normal
        normals[i1] += face_normal
        normals[i2] += face_normal
    return np.array([normalize(normal) for normal in normals], dtype=float)


def barycentric(point: tuple[int, int], a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float, float] | None:
    # Compute interpolation weights for a pixel inside a triangle.
    px, py = point
    ax, ay = a[:2]
    bx, by = b[:2]
    cx, cy = c[:2]

    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denominator) < 1e-8:
        return None

    u = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
    v = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
    w = 1.0 - u - v
    return u, v, w


def inside_frustum(ndc_points: np.ndarray) -> bool:
    # Skip triangles that are completely outside the visible clip volume.
    return np.any(
        (ndc_points[:, 0] >= -1.0)
        & (ndc_points[:, 0] <= 1.0)
        & (ndc_points[:, 1] >= -1.0)
        & (ndc_points[:, 1] <= 1.0)
        & (ndc_points[:, 2] >= -1.0)
        & (ndc_points[:, 2] <= 1.0)
    )


def shade_fragment(
    base_color: np.ndarray,
    fragment_world_position: np.ndarray,
    fragment_normal: np.ndarray,
    lights: list[Light],
) -> np.ndarray:
    # Lambertian shading: ambient light plus diffuse light from each point light.
    normal = normalize(fragment_normal)
    lit_color = base_color * AMBIENT_LIGHT

    # Each point light adds diffuse brightness based on the angle between the surface and the light.
    for light in lights:
        light_direction = normalize(light.position - fragment_world_position)
        diffuse = max(float(np.dot(normal, light_direction)), 0.0)
        lit_color += base_color * light.color * light.intensity * diffuse

    return np.clip(lit_color, 0.0, 255.0)


def render_meshes(meshes: list[Mesh], camera: Camera, lights: list[Light], width: int, height: int) -> Image.Image:
    # Color stores the final image; depth makes sure closer fragments hide farther ones.
    color_buffer = np.zeros((height, width, 3), dtype=float)
    color_buffer[:] = BACKGROUND
    depth_buffer = np.full((height, width), np.inf, dtype=float)

    # Prepare the camera pipeline once because all meshes share it.
    view_matrix = look_at(camera)
    projection_matrix = perspective(camera.fov_degrees, width / height, camera.near, camera.far)
    viewport_matrix = viewport(width, height)

    for mesh in meshes:
        # Start in model space, where each shape is defined around its own local origin.
        model_vertices = mesh.vertices
        model_normals = compute_vertex_normals(model_vertices, mesh.triangles)

        # Move vertices into world space using the mesh's model matrix.
        world_vertices_h = transform_points(mesh.model_matrix, model_vertices)
        world_vertices = world_vertices_h[:, :3] / world_vertices_h[:, 3:4]

        # Normals need a special matrix so non-uniform scaling does not distort them.
        normal_matrix = np.linalg.inv(mesh.model_matrix[:3, :3]).T
        world_normals = np.array([normalize(normal_matrix @ n) for n in model_normals], dtype=float)

        # Continue through the graphics pipeline: world -> view -> clip.
        # These stages simulate where the camera is and how a 3D lens projects depth onto a flat image.
        view_vertices_h = (view_matrix @ world_vertices_h.T).T
        clip_vertices = (projection_matrix @ view_vertices_h.T).T

        # Ignore meshes that are entirely invalid after projection.
        valid_w = np.abs(clip_vertices[:, 3]) > 1e-8
        if not np.any(valid_w):
            continue

        # Finish the pipeline: clip -> NDC -> screen pixels.
        ndc_vertices = np.zeros((clip_vertices.shape[0], 4), dtype=float)
        ndc_vertices[valid_w] = perspective_divide(clip_vertices[valid_w])
        screen_vertices_h = (viewport_matrix @ ndc_vertices.T).T
        screen_vertices = screen_vertices_h[:, :3] / screen_vertices_h[:, 3:4]

        for triangle_index, (i0, i1, i2) in enumerate(mesh.triangles):
            # Frustum culling avoids rasterizing triangles that cannot appear on screen.
            tri_ndc = np.array([ndc_vertices[i0][:3], ndc_vertices[i1][:3], ndc_vertices[i2][:3]])
            if not inside_frustum(tri_ndc):
                continue

            p0, p1, p2 = screen_vertices[i0], screen_vertices[i1], screen_vertices[i2]
            world0, world1, world2 = world_vertices[i0], world_vertices[i1], world_vertices[i2]
            normal0, normal1, normal2 = world_normals[i0], world_normals[i1], world_normals[i2]
            base_color = mesh.colors[triangle_index]

            edge1 = p1[:2] - p0[:2]
            edge2 = p2[:2] - p0[:2]
            # Back-face culling drops triangles facing away from the camera.
            signed_area = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            if signed_area <= 0:
                continue

            # Rasterize only within the triangle's bounding box for efficiency.
            min_x = max(int(math.floor(min(p0[0], p1[0], p2[0]))), 0)
            max_x = min(int(math.ceil(max(p0[0], p1[0], p2[0]))), width - 1)
            min_y = max(int(math.floor(min(p0[1], p1[1], p2[1]))), 0)
            max_y = min(int(math.ceil(max(p0[1], p1[1], p2[1]))), height - 1)

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    weights = barycentric((x, y), p0, p1, p2)
                    if weights is None:
                        continue

                    # Negative weights mean the pixel is outside the triangle.
                    u, v, w = weights
                    if u < 0 or v < 0 or w < 0:
                        continue

                    # Interpolate depth first so the z-buffer can reject hidden pixels.
                    z = u * p0[2] + v * p1[2] + w * p2[2]
                    if z >= depth_buffer[y, x]:
                        continue

                    # Interpolate the world position and normal for shading.
                    interpolated_world = u * world0 + v * world1 + w * world2
                    interpolated_normal = normalize(u * normal0 + v * normal1 + w * normal2)
                    shaded_color = shade_fragment(base_color, interpolated_world, interpolated_normal, lights)

                    # Store the winning fragment so later triangles cannot overwrite it unless they are closer.
                    depth_buffer[y, x] = z
                    color_buffer[y, x] = shaded_color

    return Image.fromarray(color_buffer.astype(np.uint8), mode="RGB")


def cube_mesh() -> tuple[np.ndarray, list[tuple[int, int, int]], np.ndarray]:
    # Define a cube with 8 corners and 12 triangles.
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    triangles = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    colors = np.array([[220, 78, 78]] * len(triangles), dtype=float)
    return vertices, triangles, colors


def pyramid_mesh() -> tuple[np.ndarray, list[tuple[int, int, int]], np.ndarray]:
    # Define a square-based pyramid with one apex.
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [-1, -1, 1],
            [0, 1.4, 0],
        ],
        dtype=float,
    )
    triangles = [
        (0, 1, 2), (0, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ]
    colors = np.array(
        [
            [186, 136, 84],
            [186, 136, 84],
            [52, 152, 219],
            [86, 177, 255],
            [46, 204, 183],
            [142, 124, 255],
        ],
        dtype=float,
    )
    return vertices, triangles, colors


def read_xyz(prompt: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    # Repeatedly ask for a 3D position until the user enters valid numeric input.
    while True:
        raw_value = input(f"{prompt} as x y z (press Enter for {default}): ").strip()
        if not raw_value:
            return default

        parts = raw_value.replace(",", " ").split()
        if len(parts) != 3:
            print("Please enter exactly 3 numbers, for example: 1.5 0 -2")
            continue

        try:
            return tuple(float(part) for part in parts)
        except ValueError:
            print("Only numeric values are allowed. Try again.")


def build_scene(
    cube_position: tuple[float, float, float],
    pyramid_position: tuple[float, float, float],
) -> tuple[list[Mesh], Camera, list[Light]]:
    # Build two objects, a camera, and two lights for the final render.
    cube_vertices, cube_triangles, cube_colors = cube_mesh()
    pyramid_vertices, pyramid_triangles, pyramid_colors = pyramid_mesh()

    # Model matrices place each object in the world with its own translation, rotation, and scale.
    cube_model = (
        translation(*cube_position)
        @ rotation_y(math.radians(30))
        @ rotation_x(math.radians(-18))
        @ scale(1.05, 1.05, 1.05)
    )
    pyramid_model = (
        translation(*pyramid_position)
        @ rotation_y(math.radians(-25))
        @ scale(1.0, 1.2, 1.0)
    )

    meshes = [
        Mesh("Cube", cube_vertices, cube_triangles, cube_colors, cube_model),
        Mesh("Pyramid", pyramid_vertices, pyramid_triangles, pyramid_colors, pyramid_model),
    ]

    camera = Camera(
        eye=np.array([3.2, 2.5, 8.0], dtype=float),
        target=np.array([0.0, 0.0, 0.0], dtype=float),
        up=np.array([0.0, 1.0, 0.0], dtype=float),
        fov_degrees=60.0,
        near=0.1,
        far=100.0,
    )

    lights = [
        Light(np.array([6.0, 8.0, 5.0], dtype=float), np.array([1.0, 0.97, 0.92], dtype=float), 1.05),
        Light(np.array([-4.0, 3.0, 2.0], dtype=float), np.array([0.55, 0.65, 1.0], dtype=float), 0.5),
    ]

    return meshes, camera, lights


def open_saved_image(output_path: Path) -> None:
    # Open the rendered image with the default image viewer on the current system.
    try:
        os.startfile(output_path)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        Image.open(output_path).show()


def main() -> None:
    # Ask the user where to place the objects, render the scene, and save the result.
    print("Enter the world coordinates for each 3D object.")
    print("A brighter background and stronger lighting are enabled to make the render easier to see.")
    cube_position = read_xyz("Cube position", (-1.8, 0.0, 0.2))
    pyramid_position = read_xyz("Pyramid position", (2.0, -0.2, -0.3))

    meshes, camera, lights = build_scene(cube_position, pyramid_position)
    image = render_meshes(meshes, camera, lights, WIDTH, HEIGHT)

    output_path = Path(__file__).with_name("linear_algebra_render.png")
    image.save(output_path)
    open_saved_image(output_path)

    print("Rendering pipeline complete.")
    print(f"Output image saved to: {output_path}")
    print("Stages used:")
    print("1. Model -> World using T * R * S homogeneous transformations")
    print("2. World -> Camera using a view matrix from eye, target, and up vectors")
    print("3. Camera -> Clip using a perspective projection matrix")
    print("4. Clip -> NDC using perspective division")
    print("5. NDC -> Screen using a viewport matrix")
    print("6. Fragment shading using barycentric interpolation and Lambertian diffuse lighting")


if __name__ == "__main__":
    main()

# Abdelrhman Ahmed Shawky 
# Hashem Hamed Elwelily 
# Osama Islam Derbalah
