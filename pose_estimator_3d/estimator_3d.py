from .model.factory import create_model
from .dataset.wild_pose_dataset import WildPoseDataset

import numpy as np
import pprint
import torch
import torch.utils.data
import yaml
from easydict import EasyDict


class Estimator3D(object):
    """Base class of 3D human pose estimator."""

    def __init__(self, config_file, checkpoint_file):
        with open(config_file, 'r') as f:
            print(f'=> Read 3D estimator config from {config_file}.')
            self.cfg = EasyDict(yaml.load(f, Loader=yaml.Loader))
            pprint.pprint(self.cfg)
        self.model = create_model(self.cfg, checkpoint_file)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f'=> Use device {self.device}.')
        self.model = self.model.to(self.device)

    def estimate(self, poses_2d, image_width, image_height):
        # pylint: disable=no-member
        dataset = WildPoseDataset(
            input_poses=poses_2d,
            seq_len=self.cfg.DATASET.SEQ_LEN,
            image_width=image_width,
            image_height=image_height
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE
        )
        poses_3d = np.zeros((poses_2d.shape[0], self.cfg.DATASET.OUT_JOINT, 3))
        frame = 0
        print('=> Begin to estimate 3D poses.')
        with torch.no_grad():
            for batch in loader:
                input_pose = batch['input_pose'].float().cuda()

                output = self.model(input_pose)
                if self.cfg.DATASET.TEST_FLIP:
                    input_lefts = self.cfg.DATASET.INPUT_LEFT_JOINTS
                    input_rights = self.cfg.DATASET.INPUT_RIGHT_JOINTS
                    output_lefts = self.cfg.DATASET.OUTPUT_LEFT_JOINTS
                    output_rights = self.cfg.DATASET.OUTPUT_RIGHT_JOINTS

                    flip_input_pose = input_pose.clone()
                    flip_input_pose[..., :, 0] *= -1
                    flip_input_pose[..., input_lefts + input_rights, :] = flip_input_pose[..., input_rights + input_lefts, :]

                    flip_output = self.model(flip_input_pose)
                    flip_output[..., :, 0] *= -1
                    flip_output[..., output_lefts + output_rights, :] = flip_output[..., output_rights + output_lefts, :]

                    output = (output + flip_output) / 2
                output[:, 0] = 0 # center the root joint
                output *= 1000 # m -> mm

                batch_size = output.shape[0]
                poses_3d[frame:frame+batch_size] = output.cpu().numpy()
                frame += batch_size
                print(f'{frame} / {poses_2d.shape[0]}')
        
        return poses_3d