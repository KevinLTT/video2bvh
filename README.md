# video2bvh

video2bvh extracts human motion in video and save it as bvh mocap file.

![demo](https://github.com/KevinLTT/video2bvh/raw/master/miscs/demo/demo.gif)

## Introduction

video2bvh consists of 3 modules: pose_estimator_2d, pose_estimator_3d and bvh_skeleton.
- **pose_estimator_2d**: Since the 3D pose estimation models we used are 2-stage model(image-> 2D pose -> 3D pose), this module is used for estimate 2D human pose (2D joint keypoint position) from image. We choose [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) as the 2d estimator. It can detect 2D joint keypoints accurately at real-time speed.
- **pose_estimator_3d**: We provide 2 models to estimate 3D human pose. 
    - [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline): This model is proposed by Julieta Martinez, Rayat Hossain, Javier Romero, and James J. Little in ICCV 2017.[[PAPER]](https://arxiv.org/pdf/1705.03098.pdf)[[CODE]](https://github.com/una-dinosauria/3d-pose-baseline). It uses single frame 2d pose as input. Its original implementation is based on TensorFlow, and we reimplemented it using PyTorch.
    - [VideoPose3D](https://github.com/facebookresearch/VideoPose3D): This model is proposed by Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli in CVPR 2019.[[PAPER]](https://arxiv.org/abs/1811.11742)[[CODE]](https://github.com/facebookresearch/VideoPose3D). It uses 2d pose sequence as input. We slightly modificate the original implementation.
- **bvh_skeleton**: This module includes the function that estimates skeleton information from 3D pose, converts 3D pose to joint angle and write motion data to bvh file.


## Dependencies
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose): See OpenPose offical [installation.md](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#python-api) for help. Note to turn on the `BUILD_PYTHON` flag while building.
- [pytorch](https://github.com/pytorch/pytorch).
- [python-opencv](https://opencv.org/).
- [numpy](https://numpy.org/)


## Pre-trained models
The original models provided by [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline) and [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) use [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 17-joint skeleton as input format (See [bvh_skeleton/h36m_skeleton.py](https://github.com/KevinLTT/video2bvh/raw/master/bvh_skeleton/h36m_skeleton.py)), but OpenPose's detection result are 25-joint (See OpenPose [output.md](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25)). So, we trained these models using 2D pose estimated by OpenPose in [Human3.6M](http://vision.imar.ro/human3.6m/description.php) dataset from scratch.

The training progress is almostly same as the originial implementation. We use subject S1, S5, S6, S7, S8 as the training set, and S9, S11 as the test set. For 3d-pose-baseline, the best MPJPE is 64.12 mm (Protocol #1), and for VideoPose3D the best MPJPE is 58.58 mm (Protocol #1). The pre-trained models can be downloaded from following links.

* [Google Drive](https://drive.google.com/drive/folders/1M2s32xQkrDhDLz-VqzvocMuoaSGR1MfX?usp=sharin)
* [Baidu Disk](https://pan.baidu.com/s/1-SRaS5FwC30-Pf_gL8bbXQ) (code: fmpz)

After you download the `models` folder, place or link it under the root directory of this project.


## Quick Start
Open [demo.ipynb](https://github.com/KevinLTT/video2bvh/raw/master/demo.ipynb) in Jupyter Notebook and follow the instructions. As you will see in the [demo.ipynb](https://github.com/KevinLTT/video2bvh/raw/master/demo.ipynb), video2bvh converts video to bvh file with 3 main steps.

### 1. Estimate 2D pose from video
<p align="center">
<img src="https://github.com/KevinLTT/video2bvh/raw/master/miscs/demo/cxk_2d_pose.gif" width="240">
</p>

### 2. Estimate 3D pose from 2D pose
<p align="center">
<img src="https://github.com/KevinLTT/video2bvh/raw/master/miscs/demo/cxk_3d_pose.gif" width="240">
</p>

### 3. Convert 3D pose to bvh motion capture file
<p align="center">
<img src="https://github.com/KevinLTT/video2bvh/raw/master/miscs/demo/cxk_bvh.gif" width="240">
</p>


## Retargeting
Once get the bvh file, you can easily retarget the motion to other 3D character model with existing tools. The girl model we used is craeted using [MakeHuman](http://www.makehumancommunity.org/), and the demo is rendered with [Blender](https://www.blender.org/). The [MakeWalk](http://www.makehumancommunity.org/wiki/Documentation:MakeWalk) plugin helps us do the retargeting work.

<p align="center">
<img src="https://github.com/KevinLTT/video2bvh/raw/master/miscs/demo/cxk_retargeting.gif" width="240">
</p>

## TODO
- [ ] Add more 2D estimators, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and [PoseResNet](https://github.com/microsoft/human-pose-estimation.pytorch).
- [ ] Smoothing 2D pose and 3D pose.
- [ ] Real-time demo.