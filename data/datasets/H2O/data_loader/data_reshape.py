import pandas as pd
import numpy as np
import os
import open3d as o3d
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional, Dict

def uniformly_sampling(d):
    '''
    Parameter
    ----------
    d:                  int, number of sampling points

    Return
    ----------
    sampling_points:    matrix of floats, dim = (8,100,3), xyz coordinates of 100 sampling points on the original 8 object meshes
    '''
    dir_ply = 'data/datasets/H2O/data/object_ply_correct'
    ply_files = os.listdir(dir_ply)
    sampling_points = np.zeros((8, d, 3))
    for i,f in enumerate(ply_files):
        path = dir_ply + "/" + f
        # Get mesh
        mesh = o3d.io.read_triangle_mesh(path)
        # Sampling
        pcd = mesh.sample_points_poisson_disk(number_of_points=d, init_factor=10)
        sampling_points[i, :, :] = np.asarray(pcd.points)
    
    return sampling_points



def transformation(sampling_points, obj_pose_rt, d=100):
    '''
    Parameter
    ----------
    sampling_points:    matrix of floats, dim = (8,100,3), xyz coordinates of 100 sampling points on the original 8 object meshes
    obj_pose_rt:        string, path of "xxxxxx.txt" file of "obj_pose_rt" in each frame
    d:                  int, number of samplinf points (equal to 100 in our case)
    
    Return
    ----------
    points_coords_3D:   matrix of floats, dim = (1, 301), label + xyz coordinates of the 100 sampling points
    '''
    # index(label) of object
    obj_rt = np.loadtxt(obj_pose_rt)[1:]
    # extrin_file = obj_pose_rt[:-23] + '/cam_pose/' + obj_pose_rt[-10:]
    # ext = np.loadtxt(extrin_file).reshape((4,4))
    obj_rt = obj_rt.reshape((4, 4))
    # rot = Rotation.from_matrix(obj_rt[0:3, 0:3])
    # euler_angle = rot.as_euler('zyx')
    obj_index = int(np.loadtxt(obj_pose_rt)[0])
    
    # Transformation of the mesh
    points_homogenous = np.vstack((np.transpose(sampling_points[obj_index-1,:,:]), np.ones((d))))
    points_coords_3D = np.dot(obj_rt, points_homogenous)[0:-1, :]
    points_coords_3D = points_coords_3D.T.reshape((-1))

    obj_index_array = np.array([np.loadtxt(obj_pose_rt)[0]])
    obj_pose = np.concatenate((obj_index_array, points_coords_3D))
    
    return obj_pose

class DataReshape:
    def __init__(
        self, 
        actions: pd.DataFrame, 
        data_dir: str, 
        type: str, 
        #config: Dict,
        #pad_type: int = 3, 
        #ifstride: bool = True
    ) -> None:
        '''
        actions: info of the actions packed in pd.DataFrame.
        data_dir: Where "action_labels, downloads, pose_lists" directories locate.
        type: Type of data, in ['train', 'val', 'test'].
        config: configs from the dataset config file.
        pad_type: Valid when action frames < frame_num. 
                  For a certain frame N after the max_action_frame, the padding below set all features:
                  PADWITHZERO: to 0.
                  PADWITHLAST: the same as the features in the max_action_frame.
                  PADBYREVERSE: the same as the features in the frame of which is "mirrored" by the max_action_frame.
                  PADBYCOPY: the same as the features in the (N-max_action_frame) frame.
                  GETMIDFRAMES: Test padding type, not in use now.
        ifstride: True if striding is applied.
        '''
        #self.config = config
        self.actions = actions
        self.frame_num = 32 #config['frame_num']
        self.data_dir = data_dir
        self.type = type
        self.n_sampling = 1000
        #self.pad_type = pad_type
        #self.ifstride = ifstride
        #self.obj_rep = config['object_representation']
        #self.n_poisson_sample = config['n_poisson_sample']
        #self.frame_selection = config["frame_selection"]
        self.sampling_points = uniformly_sampling(self.n_sampling)
        #self.sampling_points_for_repre = uniformly_sampling(self.n_poisson_sample)
        #self.obj_mesh = get_obj_mesh()
        self.samples_num = actions.shape[0]
        #self.centralize_obj = config['centralize_obj']
        #self.data_augmentation = config['data_augmentation']

    def pack_and_pad(
        self, 
        #add_contact_info: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        add_contact_info: True if contact points and distant points are added.
        '''

        # pack and pad hand_poses, object labels
        
        labels = np.zeros([self.samples_num], dtype='int64')
        hand_poses = np.zeros([self.samples_num, self.frame_num, 126]) # 126 = left hand 21 * 3 (x, y, z in order) + right hand 21 * 3
        obj_poses = np.zeros([self.samples_num, self.frame_num, 63+8+84]) # frame_length + 15: add smaller receptive field to distinguish similar actions
        
        return self.pad_by_copy(hand_poses, obj_poses, labels)
            
    

    '''
    for actions with length < required frame length, pad by copying the frames;
    for actions with length > required frame length, stride the frames uniformly
    '''
    def pad_by_copy(
        self, 
        hand_poses: np.ndarray, 
        obj_poses: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformation_matrix = np.eye(4)
        for i in range(self.samples_num):
            #WTF
            if i % 10 == 0:
                print('%d/%d actions loaded' % (i, self.samples_num))
                labels[i] = self.actions.iloc[i, 2] - 1
                start_frame = self.actions.iloc[i, 3]
                end_frame = self.actions.iloc[i, 4]
            elif self.type == 'test':
                start_frame = self.actions.iloc[i, 2]
                end_frame = self.actions.iloc[i, 3]

            hand_poses_dir = os.path.join(self.data_dir, 'data', self.actions['path'].values[i], 'cam4/hand_pose')
            obj_poses_dir = os.path.join(self.data_dir, 'data', self.actions['path'].values[i], 'cam4/obj_pose')

            cur_frame_len = end_frame - start_frame  + 1
            stride = cur_frame_len/self.frame_num
            
            for h in range(self.frame_num):
                j = np.floor(h*stride)
                hand_pose_file = os.path.join(hand_poses_dir, str(j).zfill(6) + '.txt')
                obj_pose_file = os.path.join(obj_poses_dir, str(j).zfill(6) + '.txt')
                obj_pose_rt_file = os.path.join(obj_poses_dir + '_rt', str(j).zfill(6) + '.txt')

                # get original hand pose
                hand_pose = pd.read_csv(hand_pose_file, delimiter=' ', header=None).values.flatten()
                hand_pose = np.delete(hand_pose, [0, 64, -1])
                
                
                obj_pose = pd.read_csv(obj_pose_file, delimiter=' ', header=None).values.flatten()
                obj_pose = np.delete(obj_pose, -1)


                # if contact infomation is addad
                left_hand_contact, right_hand_contact = self.add_contact_info(hand_pose, obj_pose_rt_file)
                obj_pose_final = np.concatenate((self._one_hot(obj_pose), left_hand_contact, right_hand_contact))

                hand_poses[i, h, :] = hand_pose
                obj_poses[i, h, :] = obj_pose_final

                
            
        return hand_poses, obj_poses, labels
    
    
    def add_contact_info(self, hand_pose, obj_pose_rt_file, mode='train'):
        assert mode in ['visulization', 'train']

        threshold1 = 0.02
        threshold2 = 0.2
        left_hand_pose = hand_pose[:63].reshape((21, 3))
        right_hand_pose = hand_pose[63:].reshape((21, 3))

        obj_sampling_pose = transformation(self.sampling_points, obj_pose_rt_file, self.n_sampling)
        obj_sampling_pose = np.delete(obj_sampling_pose, 0).reshape((self.n_sampling, 3))
        left_hand_obj_dist = np.amin(cdist(left_hand_pose, obj_sampling_pose), axis=1)
        right_hand_obj_dist = np.amin(cdist(right_hand_pose, obj_sampling_pose), axis=1)
        # left_hand_contact = left_hand_obj_dist
        # right_hand_contact = right_hand_obj_dist
        left_hand_contact = np.concatenate((np.multiply(left_hand_obj_dist < threshold1, 0.1), np.multiply(left_hand_obj_dist > threshold2, 0.1)))
        right_hand_contact = np.concatenate((np.multiply(right_hand_obj_dist < threshold1, 0.1), np.multiply(right_hand_obj_dist > threshold2, 0.1)))

        # obj_sampling_pose = transformation(self.sampling_points1, obj_pose_rt_file, 32)
        # obj_sampling_pose = np.delete(obj_sampling_pose, 0).reshape((500, 3))
        # obj_left_hand_dist = np.amin(cdist(left_hand_pose, obj_sampling_pose), axis=0)
        # obj_right_hand_dist = np.amin(cdist(right_hand_pose, obj_sampling_pose), axis=0)

        return left_hand_contact, right_hand_contact
    
    

    def _one_hot(self, obj_pose):
        obj_dummy = np.zeros([8])
        obj_dummy[int(obj_pose[0])-1] = 0.1 # for normalization purpose
        obj_pose = np.delete(obj_pose, 0)
        obj_pose = np.concatenate((obj_dummy, obj_pose))
        return obj_pose