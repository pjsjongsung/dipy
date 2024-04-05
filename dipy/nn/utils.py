import numpy as np
from scipy.ndimage import affine_transform, label
from dipy.align.reslice import reslice


def normalize(image, min_v=None, max_v=None, new_min=-1, new_max=1):
    r"""
    normalization function

    Parameters
    ----------
    image : np.ndarray
    min_v : int or float, optional
        minimum value range for normalization
        intensities below min_v will be clipped
        if None it is set to min value of image
        Default : None
    max_v : int or float, optional
        maximum value range for normalization
        intensities above max_v will be clipped
        if None it is set to max value of image
        Default : None
    new_min : int or float, optional
        new minimum value after normalization
        Default : 0
    new_max : int or float, optional
        new maximum value after normalization
        Default : 1

    Returns
    -------
    np.ndarray
        Normalized image from range new_min to new_max
    """
    if min_v is None:
        min_v = np.min(image)
    if max_v is None:
        max_v = np.max(image)
    return np.interp(image, (min_v, max_v), (new_min, new_max))


def unnormalize(image, norm_min, norm_max, min_v, max_v):
    r"""
    unnormalization function

    Parameters
    ----------
    image : np.ndarray
    norm_min : int or float
        minimum value of normalized image
    norm_max : int or float
        maximum value of normalized image
    min_v : int or float
        minimum value of unnormalized image
    max_v : int or float
        maximum value of unnormalized image

    Returns
    -------
    np.ndarray
        unnormalized image from range min_v to max_v
    """
    return (image - norm_min) / (norm_max-norm_min) * \
           (max_v - min_v) + min_v


def set_logger_level(log_level, logger):
    """ Change the logger to one of the following:
    DEBUG, INFO, WARNING, CRITICAL, ERROR

    Parameters
    ----------
    log_level : str
        Log level for the logger
    """
    logger.setLevel(level=log_level)


def calculate_output_bounds_and_offset(shape, affine_matrix, volume=None,
                                       considered_points='corners'):
    """ Calculate the necessary output volume bounds and offset to ensure the entire
    transformed volume is captured without negative coordinates.

    Parameters
    ----------
    volume : np.ndarray
        The volume we are calculating the bounds of
        Note that it is only used if we set considered_points
        to 'non_zeros'
    shape : list, tuple or numpy array (3,)
        The shape of the volume minus the channel if there is any
    affine_matrix : np.ndarray (4, 4)
        The affine matrix provided with the volume from the nifti file
    considered_points : str
        Which points to consider when calculating the boundaries
        The options are 'corners', 'all', 'non_zeros'
    """
    # Define all corner points of the volume
    if considered_points == 'corners':
        corners = np.array([
            [0, 0, 0, 1],
            [shape[0]-1, 0, 0, 1],
            [0, shape[1]-1, 0, 1],
            [0, 0, shape[2]-1, 1],
            [shape[0]-1, shape[1]-1, shape[2]-1, 1],
            [shape[0]-1, 0, shape[2]-1, 1],
            [0, shape[1]-1, shape[2]-1, 1],
            [shape[0]-1, shape[1]-1, 0, 1]
        ])
    elif considered_points == 'non_zeros':
        if volume is None:
            raise ValueError("volume must be provided if considered_points" +
                             " is non_zeros")
        if len(shape) == 4:
            corners = np.argwhere(np.all([volume[..., c]!=0
                                          for c in range(shape[-1])], axis=0))
        else:
            corners = np.argwhere(volume!=0)
        corners = np.concatenate([corners,
                                  np.ones((len(corners), 1))], axis=-1)
    elif considered_points == 'all':
        corners = np.reshape(np.moveaxis(np.indices(shape[:3]), 0, -1),
                             (shape[0]*shape[1]*shape[2], 3)) 
    else:
        raise ValueError("corners, all, non_zeros are the only supported" +
                         " input for considered_points. Got " +
                         str(considered_points))

    # Transform the corners to find the new bounds
    inv_affine_matrix = np.linalg.inv(affine_matrix)
    transformed_corners = (inv_affine_matrix @ corners.T).T
    min_bounds = transformed_corners.min(axis=0)[:3]
    max_bounds = transformed_corners.max(axis=0)[:3]

    # Calculate the required offset to ensure all coordinates are positive
    offset = -min_bounds
    new_shape = np.ceil(max_bounds + offset).astype(int)

    new_affine = inv_affine_matrix.copy()
    new_affine[:3, 3] += offset
    new_affine = np.linalg.inv(new_affine)
    
    return new_shape, new_affine


def transform_img(image, affine, voxsize=None,
                  init_shape=(256, 256, 256), scale=2):
    r"""
    Function to reshape image as an input to the model

    Parameters
    ----------
    image : np.ndarray
        Image to transform to voxelspace
    affine : np.ndarray
        Affine matrix provided by the file
    voxsize : np.ndarray (3,), optional
        Voxel size of the image
    init_shape : list, tuple or numpy array (3,)
        Initial shape to transform the image to
    scale : float, optional
        How much we want to scale the image
        Default is 2

    Returns
    -------
    transformed_img : np.ndarray
    """
    if voxsize is not None and np.any(voxsize != np.ones(3)):
        image, affine2 = reslice(image, affine, voxsize, (1, 1, 1))
    else:
        affine2 = affine.copy()

    inv_affine = np.linalg.inv(affine2)
    ori_shape = image.shape

    new_shape, new_affine = \
        calculate_output_bounds_and_offset(ori_shape, inv_affine)
    
    if init_shape is None:
        init_shape = new_shape
    else:
        init_shape = np.array(init_shape)
        pad_crop_v = new_shape - init_shape
        for d in range(3):
            if pad_crop_v[d] < 0:
                new_affine[:3, d] += pad_crop_v[d] / 2
            elif pad_crop_v[d] > 0:
                new_affine[:3, d] -= pad_crop_v[d] / 2

    
    if len(ori_shape) == 4:
        transformed_img = np.zeros((*init_shape, ori_shape[-1]), dtype=image.dtype)

        for c in range(ori_shape[-1]):
            affine_transform(
                image[..., c],
                new_affine[:3, :3],
                offset=new_affine[:3, 3],
                output_shape=init_shape,
                output=transformed_img[..., c],
                order=1
            )
    else:
        transformed_img = np.zeros(init_shape, dtype=image.dtype)

        affine_transform(
            image,
            new_affine[:3, :3],
            offset=new_affine[:3, 3],
            output_shape=init_shape,
            output=transformed_img,
            order=1
        )

    transformed_img, _ = reslice(transformed_img, np.eye(4), (1, 1, 1),
                                 (scale, scale, scale))
    return transformed_img, np.stack([affine2, new_affine], axis=0), ori_shape


def recover_img(image, affines, ori_shape, image_shape,
                init_shape=(256, 256, 256), voxsize=None, scale=2):
    r"""
    Function to recover image back to its original shape
    Meant to be used with transform_img

    Parameters
    ----------
    image : np.ndarray
        Image to recover
    affines : np.ndarray (2, 4, 4)
        Affine matrices provided from transform_img
    ori_shape : np.ndarray (3,)
        Original shape of isotropic image
    image_shape : tuple (3,)
        Original shape of actual image
    init_shape : list, tuple or numpy array (3,)
        Initial shape to transform the image to
        Should be equal to the one used in transform_img
    voxsize : np.ndarray (3,), optional
        Voxel size of the original image
    scale : float, optional
        Scale that was used in transform_img

    Returns
    -------
    recovered_img : np.ndarray
    """
    image, _ = reslice(image, np.eye(4), (scale, scale, scale), (1, 1, 1))
    inv_affine = np.linalg.inv(affines[1])
    recovered_img = np.zeros(ori_shape, dtype=image.dtype)
    if len(ori_shape) == 4:
        for c in range(ori_shape[-1]):
            affine_transform(
                image[..., c],
                inv_affine[:3, :3],
                offset=inv_affine[:3, 3],
                output_shape=ori_shape,
                output=recovered_img[..., c],
                order=1
            )
    else:
        affine_transform(
            image,
            inv_affine[:3, :3],
            offset=inv_affine[:3, 3],
            output_shape=ori_shape,
            output=recovered_img,
            order=1
        )
    
    return recovered_img


def correct_minor_errors(binary_img):
    """
    Remove any small mask chunks or holes
    that could be in the segmentation output.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image

    Returns
    -------
    largest_img : np.ndarray
    """
    largest_img = np.zeros_like(binary_img)
    chunks, n_chunk = label(np.abs(1-binary_img))
    u, c = np.unique(chunks[chunks != 0], return_counts=True)
    target = u[np.argmax(c)]
    largest_img = np.where(chunks == target, 0, 1)

    chunks, n_chunk = label(largest_img)
    u, c = np.unique(chunks[chunks != 0], return_counts=True)
    target = u[np.argmax(c)]
    largest_img = np.where(chunks == target, 1, 0)

    for x in range(largest_img.shape[0]):
        chunks, n_chunk = label(np.abs(1-largest_img[x]))
        u, c = np.unique(chunks[chunks != 0], return_counts=True)
        target = u[np.argmax(c)]
        largest_img[x] = np.where(chunks == target, 0, 1)
    for y in range(largest_img.shape[1]):
        chunks, n_chunk = label(np.abs(1-largest_img[:, y]))
        u, c = np.unique(chunks[chunks != 0], return_counts=True)
        target = u[np.argmax(c)]
        largest_img[:, y] = np.where(chunks == target, 0, 1)
    for z in range(largest_img.shape[2]):
        chunks, n_chunk = label(np.abs(1-largest_img[..., z]))
        u, c = np.unique(chunks[chunks != 0], return_counts=True)
        target = u[np.argmax(c)]
        largest_img[..., z] = np.where(chunks == target, 0, 1)

    return largest_img
