import torch
import yaml
from easydict import EasyDict


def create_model(cfg, checkpoint_file):
    if cfg.MODEL.NAME == 'linear_model':
        from .linear_model import LinearModel
        model = LinearModel(
            in_joint=cfg.DATASET.IN_JOINT,
            in_channel=cfg.DATASET.IN_CHANNEL,
            out_joint=cfg.DATASET.OUT_JOINT,
            out_channel=cfg.DATASET.OUT_CHANNEL,
            block_num=cfg.MODEL.BLOCK_NUM,
            hidden_size=cfg.MODEL.HIDDEN_SIZE,
            dropout=cfg.MODEL.DROPOUT,
            bias=cfg.MODEL.BIAS,
            residual=cfg.MODEL.RESIDUAL
        )
    elif cfg.MODEL.NAME == 'video_pose':
        from .video_pose import VideoPose
        model = VideoPose(
            in_joint=cfg.DATASET.IN_JOINT,
            in_channel=cfg.DATASET.IN_CHANNEL,
            out_joint=cfg.DATASET.OUT_JOINT,
            out_channel=cfg.DATASET.OUT_CHANNEL,
            filter_widths=cfg.MODEL.FILTER_WIDTHS,
            hidden_size=cfg.MODEL.HIDDEN_SIZE,
            dropout=cfg.MODEL.DROPOUT,
            dsc=cfg.MODEL.DSC   
        )
    else:
        raise ValueError(f'Model name {cfg.MODEL.NAME} is invalid.')

    print(f'=> Load checkpoint {checkpoint_file}')
    pretrained_dict = torch.load(checkpoint_file)['model_state']
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.eval()

    return model
