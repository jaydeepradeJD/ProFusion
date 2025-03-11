import numpy as np
from pytorch3d.renderer import PerspectiveCameras, OrthographicCameras, FoVOrthographicCameras
import torch

def pixel_to_ndc(x, y, height, width):
    """ Converts Pixel (screen-space) coordinates to NDC coordinates

    The convention used is as follows:
        * Screen Space - X: [0, W-1], Y: [0, H-1]
        * NDC Space - X: [1, -1], Y: [1, -1]

    where (0, 0) and (H, W) are the top-left and bottom right corners of the image in the screen space (pixel coordinates), and, (1, 1) and (-1,
    -1) are the top-left and the bottom-right corners in the NDC space (NDC coordinates) respectively.

    Note that the output of this function is a numpy array (loses differentiability).

    Args:
        x(int|float|list|np.array): A scalar or a list (or array) of values indicating the x-coordinate(s) in the screen space
        y(int|float|list|np.array): A scalar or a list (or array) of values indicating the y-coordinate(s) in the screen space
        height(int): The height of the image in screen space (pixels)
        width(int): The width of the image in screen space (pixels)

    Returns:
        tuple[np.arraym, np.array]: A tuple containing:
                                        1) x_ndc(np.ndarray): The NDC coordinates corresponding to the provided x-coordinates (pixel coords.),
                                        2) y_ndc(np.ndarray): The NDC coordinates corresponding to the provided y-coordinates (pixel coords.)

    """
    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x], dtype=np.int32)
    elif isinstance(x, list):
        x = np.array(x, dtype=np.int32)

    if isinstance(y, float) or isinstance(y, int):
        y = np.array([y], dtype=np.int32)
    elif isinstance(y, list):
        y = np.array(y, dtype=np.int32)

    x_ndc = np.linspace(1, -1, width, dtype=np.float32)[x]
    y_ndc = np.linspace(1, -1, height, dtype=np.float32)[y]
    return x_ndc, y_ndc


def create_cameras_with_identity_extrinsics(num_cameras, R=None, T=None, min_x=-1.1, max_x=1.1, min_y=-1.1, max_y=1.1, znear=-1.1, zfar=1.1, image_size=[256, 256]):

    cameras = []
    image_size_ = torch.tensor([image_size])
    if R is None:
        R = torch.eye(3, dtype=torch.float32)
    if T is None:
        T = torch.zeros((3), dtype=torch.float32)
    
    for i in range(num_cameras):
        camera = FoVOrthographicCameras(
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            znear=znear,
            zfar=zfar
        )
        cameras.append(camera)

    return cameras

def get_queryCameras(device, min_x=-1.1, max_x=1.1, min_y=-1.1, max_y=1.1, znear=-1.1, zfar=1.1, R=None, T=None,  image_size=[256, 256]):
    query_cameras = []
    image_size_ = torch.tensor([image_size])
    for i in range(R.shape[0]):
        query_camera = FoVOrthographicCameras(
            R=R[i].unsqueeze(0),
            T=T[i].unsqueeze(0),
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            znear=znear,
            zfar=zfar,
            device=device
        )
        query_cameras.append(query_camera)

    return query_cameras
