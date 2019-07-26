import numpy as np
import torch


class WildPoseDataset(object):

    def __init__(self, input_poses, seq_len, image_width, image_height):
        self.seq_len = seq_len
        self.input_poses = normalize_screen_coordiantes(input_poses, image_width, image_height)


    def __len__(self):
        return self.input_poses.shape[0]


    def __getitem__(self, idx):
        frame = idx
        start = frame - self.seq_len//2
        end = frame + self.seq_len//2 + 1
        
        valid_start = max(0, start)
        valid_end = min(self.input_poses.shape[0], end)
        pad = (valid_start - start, end - valid_end)
        input_pose = self.input_poses[valid_start:valid_end]
        if pad != (0, 0):
            input_pose = np.pad(input_pose, (pad, (0, 0), (0, 0)), 'edge')
        if input_pose.shape[0] == 1:
            # squeeze time dimension if sequence length is 1
            input_pose = np.squeeze(input_pose, axis=0)

        sample = { 'input_pose': input_pose }
        return sample


def normalize_screen_coordiantes(pose, w, h):
    """
    Args:
        pose: numpy array with shape (joint, 2).
    Return:
        normalized pose that [0, WIDTH] is maped to [-1, 1] while preserving the aspect ratio.
    """
    assert pose.shape[-1] == 2
    return pose/w*2 - [1, h/w]


def flip_pose(pose, lefts, rights):
    if isinstance(pose, np.ndarray):
        p = pose.copy()
    elif isinstance(pose, torch.Tensor):
        p = pose.clone()
    else:
        raise TypeError(f'{type(pose)}')

    p[..., 0] *= -1
    p[..., lefts + rights, :] = p[..., rights + lefts, :]
    return p
