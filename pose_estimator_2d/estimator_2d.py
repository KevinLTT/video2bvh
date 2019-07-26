import abc

class Estimator2D(object):
    """Base class of 2D human pose estimator."""

    def __init__(self):
        pass

    @abc.abstractclassmethod
    def estimate(self, img_list, bbox_list=None):
        """
        Args:
            img_list: List of image read by opencv(channel order BGR).
            bbox_list: List of bounding-box (left_top x, left_top y, 
                bbox_width, bbox_height).
        Return:
            keypoints_list: List of keypoint position (joint_num, x, y,
            confidence)
        """
        pass