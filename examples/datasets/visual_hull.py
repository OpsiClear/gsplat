import numpy as np
from numba import njit
from scipy import ndimage
from tqdm import tqdm


@njit
def project_points(points, camera_matrix):
    """
    Project 3D points using camera matrix.

    Args:
        points: Nx4 array of homogeneous 3D points
        camera_matrix: 3x4 projection matrix
    Returns:
        points_2d: Nx2 array of 2D points
        valid_points: N boolean array
    """
    n_points = len(points)
    points_2d = np.empty((n_points, 2), dtype=np.float64)
    valid_points = np.empty(n_points, dtype=np.bool_)

    # Process points in batches for better cache utilization
    batch_size = 1024
    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch = points[start:end]

        # Project batch
        proj = np.empty((end - start, 3))
        for i in range(end - start):
            for j in range(3):
                acc = 0.0
                for k in range(4):
                    acc += camera_matrix[j, k] * batch[i, k]
                proj[i, j] = acc

        # Process projected points
        for i in range(end - start):
            if proj[i, 2] > 0:
                points_2d[start + i, 0] = proj[i, 0] / proj[i, 2]
                points_2d[start + i, 1] = proj[i, 1] / proj[i, 2]
                valid_points[start + i] = True
            else:
                valid_points[start + i] = False

    return points_2d, valid_points


@njit
def process_points_batch(points_2d, mask, valid_points, grid_mask):
    """
    Process a batch of 2D points against the mask image.

    Args:
        points_2d: Nx2 array of 2D points
        mask: HxW binary mask image
        valid_points: N boolean array indicating valid points
        grid_mask: N boolean array to store results
    """
    h, w = mask.shape
    n_points = len(points_2d)

    # Pre-compute coordinates for better memory locality
    coords = np.empty((n_points, 2), dtype=np.int32)
    for i in range(n_points):
        coords[i, 0] = int(round(points_2d[i, 0]))
        coords[i, 1] = int(round(points_2d[i, 1]))

    # Process points with better bounds checking
    for i in range(n_points):
        if valid_points[i]:
            x, y = coords[i]
            if 0 <= x < w and 0 <= y < h:
                grid_mask[i] = mask[y, x]
            else:
                grid_mask[i] = False


@njit
def initialize_voxel_grid(min_bound, max_bound, grid_shape):
    """
    Initialize the voxel grid points in compiled code.
    """
    nx, ny, nz = grid_shape
    total_points = nx * ny * nz
    points = np.zeros((total_points, 4))
    idx = 0

    # Calculate step sizes for each axis to handle non-cubic grids
    # Avoid division by zero if an axis has size 1
    step_x = (max_bound[0] - min_bound[0]) / (nx - 1) if nx > 1 else 0.0
    step_y = (max_bound[1] - min_bound[1]) / (ny - 1) if ny > 1 else 0.0
    step_z = (max_bound[2] - min_bound[2]) / (nz - 1) if nz > 1 else 0.0

    for i in range(nx):
        x = min_bound[0] + i * step_x
        for j in range(ny):
            y = min_bound[1] + j * step_y
            for k in range(nz):
                z = min_bound[2] + k * step_z
                points[idx] = [x, y, z, 1.0]
                idx += 1

    return points


@njit
def combine_grids_forgiving(grid_mask, count_grid, shape):
    """
    Track number of views each point appears in instead of strict intersection.

    Args:
        grid_mask: Current view's occupancy
        count_grid: Running count of views each point appears in
        shape: Shape of the grid
    """
    view_grid = grid_mask.reshape(shape)

    # Process in blocks for better cache utilization
    block_size = 64
    result = np.zeros(count_grid.shape, dtype=np.int32)

    for i in range(0, shape[0], block_size):
        i_end = min(i + block_size, shape[0])
        for j in range(0, shape[1], block_size):
            j_end = min(j + block_size, shape[1])
            for k in range(0, shape[2], block_size):
                k_end = min(k + block_size, shape[2])

                # Add 1 to count where point appears in current view
                result[i:i_end, j:j_end, k:k_end] = (
                    count_grid[i:i_end, j:j_end, k:k_end]
                    + view_grid[i:i_end, j:j_end, k:k_end]
                )

    return result


class VisualHull:
    def __init__(self, voxel_size=128, bounds=(-1, 1)):
        # Previous initialization code remains the same
        self.voxel_size = voxel_size
        if isinstance(bounds, tuple) and len(bounds) == 2:
            self.min_bound = (
                np.array(bounds[0])
                if isinstance(bounds[0], (list, np.ndarray))
                else np.array([bounds[0]] * 3)
            )
            self.max_bound = (
                np.array(bounds[1])
                if isinstance(bounds[1], (list, np.ndarray))
                else np.array([bounds[1]] * 3)
            )
        else:
            self.min_bound = np.array(bounds[0])
            self.max_bound = np.array(bounds[1])

        # Calculate grid shape to make voxels approximately cubic
        extent = self.max_bound - self.min_bound
        # Handle cases where an extent is zero to avoid division by zero
        extent[extent == 0] = 1e-6
        longest_axis_length = np.max(extent)

        # The size of one voxel's side
        voxel_step_size = longest_axis_length / (voxel_size - 1)

        # Determine grid shape based on the step size
        grid_shape = np.ceil(extent / voxel_step_size).astype(int)
        # Ensure shape is at least 1 in each dimension
        grid_shape = np.maximum(grid_shape, 1)

        # Recalculate max_bound to be an even multiple of the grid steps
        self.max_bound = self.min_bound + (grid_shape - 1) * voxel_step_size

        # Initialize empty 3D occupancy grid
        self.grid = np.zeros(grid_shape, dtype=bool)

        # Initialize points
        self.points = initialize_voxel_grid(self.min_bound, self.max_bound, grid_shape)

        # Track whether this is the first view
        self.is_first_view = True

        # Cache the shape tuple
        self._shape = tuple(grid_shape)

        # Pre-allocate reusable arrays
        self._view_grid = np.zeros(self._shape, dtype=bool)
        self._grid_mask = np.zeros(len(self.points), dtype=bool)

        # Add count grid to track views
        self._count_grid = np.zeros(self._shape, dtype=np.int32)
        self._total_views = 0

    def add_from_view(self, camera_matrix, mask):
        """Update visual hull with view counting."""
        # Project points
        points_2d, valid_points = project_points(self.points, camera_matrix)

        # Reset grid mask
        self._grid_mask.fill(False)

        # Process points
        if np.any(valid_points):
            process_points_batch(points_2d, mask, valid_points, self._grid_mask)

        # Update count grid
        self._count_grid = combine_grids_forgiving(
            self._grid_mask,
            self._count_grid,
            self._shape,
        )

        self._total_views += 1
        self.is_first_view = False

    def finalize_grid(self):
        """Convert count grid to final occupancy grid using N-1 threshold."""
        # Points must appear in at least total_views - 1 views
        threshold = 0.90 * self._total_views
        self.grid = self._count_grid >= threshold
        return self.grid

    def process_all_views(self, camera_matrices, masks):
        """Process all views and finalize grid."""
        self.is_first_view = True
        self._total_views = 0
        self._count_grid.fill(0)

        for cam_mat, mask in tqdm(zip(camera_matrices, masks), desc="Carving Visual Hull"):
            self.add_from_view(cam_mat, mask)

        self.finalize_grid()



    def get_surface_points(self):
        """
        Return the 3D coordinates of voxels on the surface of the hull.
        The surface is calculated as the difference between the hull and its erosion.
        """
        if not self.grid.any():
            print("Warning: Visual hull grid is empty. Cannot extract surface.")
            return np.empty((0, 3), dtype=np.float32)

        # Erode the grid to find interior voxels
        eroded_grid = ndimage.binary_erosion(self.grid, iterations=2)

        # The surface is the set of voxels in the original grid minus the eroded grid
        surface_grid = self.grid & ~eroded_grid

        # The order of points corresponds to the flattened grid
        surface_mask = surface_grid.flatten()
        return self.points[surface_mask, :3]

    def get_occupied_points(self):
        """
        Return the 3D coordinates of occupied voxels.

        Returns:
            numpy array: Nx3 array of point coordinates
        """
        # The order of points corresponds to the flattened grid
        occupied_mask = self.grid.flatten()
        return self.points[occupied_mask, :3]

    def save_to_file(self, filename):
        """Save the visual hull grid and bounds to a binary file."""
        np.savez(
            filename,
            grid=self.grid,
            min_bound=self.min_bound,
            max_bound=self.max_bound,
        )

    @classmethod
    def load_from_file(cls, filename, voxel_size=128, bounds=(-1, 1)):
        """Load a visual hull from a binary file."""
        hull = cls(voxel_size, bounds)
        data = np.load(filename)
        hull.grid = data["grid"]
        hull.min_bound = data["min_bound"]
        hull.max_bound = data["max_bound"]
        return hull
