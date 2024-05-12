import numpy as np
from numpy.typing import NDArray

def to_hom_3d(points: NDArray) -> NDArray:
    """
    Convert the input points to homogeneous coordinates.

    Args:
    -----
    points: NDArray (N, 3)
        Input points in cartesian coordinates
    
    Returns:
    --------
    points_hom: NDArray (N, 4)
        Homogeneous coordinates of the input points
    """
    assert len(points.shape) == 2 and points.shape[1] == 3 , \
        f"points should be of shape (N, 3) but got {points.shape}"
    return np.hstack((points, np.ones((points.shape[0], 1))))

def to_cart_3d(points_hom: NDArray) -> NDArray:
    """
    Convert the input points in homogeneous coordinates to cartesian coordinates.

    Args:
    -----
    points_hom: NDArray (N, 4)
        Input points in homogeneous coordinates
    
    Returns:
    --------
    points: NDArray (N, 3)
        Cartesian coordinates of the input points
    """
    assert len(points_hom.shape) == 2 and points_hom.shape[1] == 4, \
        f"points_hom should be of shape (N, 4) but got {points_hom.shape}"
    return points_hom[:, :3] / points_hom[:, 3, None]

def invert_transformation_matrix(T: NDArray) -> NDArray:
    """
    Invert the transformation matrix T

    Args:
    -----
    T: NDArray (4, 4)
        Transformation matrix

    Returns:
    --------
    T_inv: NDArray (4, 4)
        Inverted transformation matrix
    """
    assert T.shape == (4, 4), f"T should be of shape (4, 4) but got {T.shape}"

    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv

def transform_pts_hom(pts_hom: NDArray, T: NDArray) -> NDArray:
    """
    Transform the input points using the input transformation matrix.

    Args:
    -----
    pts_hom: NDArray (N, 4)
        Input points in homogeneous coordinates
    T: NDArray (4, 4)
        Transformation matrix
    
    Returns:
    --------
    pts_transformed_hom: NDArray (N, 4)
        Transformed points in homogeneous coordinates
    """
    assert len(pts_hom.shape) == 2 and pts_hom.shape[1] == 4, \
        f"pts should be of shape (N, 4) but got {pts_hom.shape}"
    assert T.shape == (4, 4), f"T should be of shape (4, 4) but got {T.shape}"

    return np.dot(T, pts_hom.T).T

def transfrom_pts_cart(pts_cart: NDArray, T: NDArray) -> NDArray:
    """
    Transform the input points using the input transformation matrix.

    Args:
    -----
    pts_cart: NDArray (N, 4)
        Input points in homogeneous coordinates
    T: NDArray (4, 4)
        Transformation matrix
    
    Returns:
    --------
    pts_transformed_hom: NDArray (N, 4)
        Transformed points in homogeneous coordinates
    """
    assert len(pts_cart.shape) == 2 and pts_cart.shape[1] == 3, \
        f"pts should be of shape (N, 4) but got {pts_cart.shape}"
    assert T.shape == (4, 4), f"T should be of shape (4, 4) but got {T.shape}"

    pts_hom = to_hom_3d(pts_cart)
    pts_transformed_hom = transform_pts_hom(pts_hom, T)
    return to_cart_3d(pts_transformed_hom)
