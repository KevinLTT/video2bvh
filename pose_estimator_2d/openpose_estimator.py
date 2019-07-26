from .estimator_2d import Estimator2D
from openpose import pyopenpose as op


class OpenPoseEstimator(Estimator2D):

    def __init__(self, model_folder):
        """
        OpenPose 2D pose estimator. See [https://github.com/
        CMU-Perceptual-Computing-Lab/openpose/tree/ master/examples/
        tutorial_api_python] for help.
        Args: 
        """
        super().__init__()
        params = {'model_folder': model_folder, 'render_pose': 0}
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def estimate(self, img_list, bbox_list=None):
        """See base class."""
        keypoints_list = []
        for i, img in enumerate(img_list):
            if bbox_list:
                x, y, w, h = bbox_list[i]
                img = img[y:y+h, x:x+w]
            datum = op.Datum()
            datum.cvInputData = img
            self.opWrapper.emplaceAndPop([datum])
            keypoints = datum.poseKeypoints
            if bbox_list:
                # TODO: restore coordinate
                pass
            keypoints_list.append(datum.poseKeypoints)

        return keypoints_list