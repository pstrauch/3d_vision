import numpy as np
from numpy.typing import NDArray
from typing import List

def format_hand_poses(hand_poses: NDArray) -> NDArray:
    """
    Format the raw hand_poses vector into a 21x3 matrix for each hand

    Args:
    -----
    hand_poses: NDArray (128,)
        Raw hand poses vector

    Returns:
    --------
    hand_poses_left: NDArray (21, 3)
        Hand poses matrix for left hand
    hand_poses_right: NDArray (21, 3)
        Hand poses matrix for right hand
    """
    assert hand_poses.shape == (128,), f"hand_poses should be of shape (128,) but got {hand_poses.shape}"

    hand_poses_left = hand_poses[1:64].reshape((21, 3))
    hand_poses_right = hand_poses[65:].reshape((21, 3))

    return hand_poses_left, hand_poses_right

def format_object_poses(obj_poses: NDArray) -> List[NDArray]:
    """
    Format and group the raw obj_poses vector into a list of the poses.

    The list has 3 elements:
    1. (3,)   NDArray of the object's center pose
    2. (8,3)  NDArray of the object's corner vertices
    3. (12,3) NDArray of the object's mid-edge vertices

    Args:
    -----
    obj_poses: NDArray (64,)
        Raw object poses vector

    Returns:
    --------
    obj_poses_list: List[NDArray]
        List of grouped object poses
    """
    assert obj_poses.shape == (64,), f"obj_poses should be of shape (64,) but got {obj_poses.shape}"

    obj_center = obj_poses[1:4]
    obj_corners = obj_poses[4:28].reshape((8, 3))
    obj_edges = obj_poses[28:].reshape((12, 3))

    return [obj_center, obj_corners, obj_edges]