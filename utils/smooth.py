def filter_missing_value(keypoints_list, method='ignore'):
    # TODO: impletemd 'interpolate' method.
    """Filter missing value in pose list.
    Args:
        keypoints_list: Estimate result returned by 2d estimator. Missing value 
        will be None.
        method: 'ignore' -> drop missing value.
    Return:
        Keypoints list without missing value.
    """

    result = []
    if method == 'ignore':
        result = [pose for pose in keypoints_list if pose is not None]
    else:
        raise ValueError(f'{method} is not a valid method.')

    return result