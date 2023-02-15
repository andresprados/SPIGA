import numpy as np


def affine2homogeneous(points):
    '''Returns the points completed with a new last coordinate
    equal to 1
    Arguments
    ---------
    points: np.array of shape (num_points, dim)
    Returns
    -------
    hpoints: np.array of shape (num_points, dim + 1),
        of the points completed with ones'''

    num_points = points.shape[0]
    hpoints = np.hstack(
        (points, np.repeat(1, num_points).reshape(num_points, 1)))
    return hpoints


def get_similarity_matrix(deg_angle, scale, center):
    '''Similarity matrix.
    Arguments:
    ---------
    deg_angle: rotation angle in degrees
    scale: factor scale
    center: coordinates of the rotation center

    Returns:
    -------
    matrix: (2, 3) numpy array representing the
        similarity matrix.
    '''
    x0, y0 = center
    angle = np.radians(deg_angle)

    matrix = np.zeros((2, 3))
    matrix[0:2, 0:2] = [[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]]
    matrix[0: 2, 0: 2] *= scale

    matrix[:, 2] = [(1 - scale * np.cos(angle)) * x0 +
                    scale * np.sin(angle) * y0,
                    -scale * np.sin(angle) * x0 +
                    (1 - scale * np.cos(angle)) * y0]
    return matrix


def get_inverse_similarity_matrix(deg_angle, scale, center):
    '''Returns the inverse of the affine similarity
    Arguments
    ---------
    deg_angle: angle in degrees of the rotation
    center: iterable of two components (x0, y0),
            center of the rotation
    scale: float, scale factor
    Returns
    -------
    matrix: np.array of shape (2, 3) with the coordinates of
    the inverse of the similarity'''

    x0, y0 = center
    angle = np.radians(deg_angle)
    inv_scale = 1 / scale
    matrix = np.zeros((2, 3))
    matrix[0:2, 0:2] = [[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]]
    matrix[0:2, 0:2] *= inv_scale

    matrix[:, 2] = [(1 - inv_scale * np.cos(angle)) * x0 -
                    inv_scale * np.sin(angle) * y0,
                    inv_scale * np.sin(angle) * x0 +
                    (1 - inv_scale * np.cos(angle)) * y0]

    return matrix


def get_inverse_transf(affine_transf):
    A = affine_transf[0:2, 0:2]
    b = affine_transf[:, 2]

    inv_A = np.linalg.inv(A)  # we assume A invertible!

    inv_affine = np.zeros((2, 3))
    inv_affine[0:2, 0:2] = inv_A
    inv_affine[:, 2] = -inv_A.dot(b)

    return inv_affine


def image2vect(image):
    '''
    Input:
    image[batch_size, num_channels, im_size_x, im_size_y]
    Output:
    vect[batch_size, num_channels, im_size_x*im_size_y]
    '''
    vect = image.reshape(*image.shape[0:-2], -1)
    return vect


def rotation_matrix_to_euler(rot_matrix):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
    a00, a01, a02 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    a10, a11, a12 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    a20, a21, a22 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]
    if abs(1.0 - a10) <= np.finfo(float).eps:  # singularity at north pole / special case a10 == 1
        yaw = np.arctan2(a02, a22)
        pitch = np.pi/2.0
        roll = 0
    elif abs(-1.0 - a10) <= np.finfo(float).eps:  # singularity at south pole / special case a10 == -1
        yaw = np.arctan2(a02, a22)
        pitch = -np.pi/2.0
        roll = 0
    else:  # standard case
        yaw = np.arctan2(-a20, a00)
        pitch = np.arcsin(a10)
        roll = np.arctan2(-a12, a11)
    # Convert to degrees
    euler = np.array([yaw, pitch, roll])*(180.0/np.pi)
    # Change coordinates system
    euler = np.array([(-euler[0])+90, -euler[1], (-euler[2])-90])
    if euler[0] > 180: euler[0] -= 360
    elif euler[0] < -180: euler[0] += 360
    if euler[1] > 180: euler[1] -= 360
    elif euler[1] < -180: euler[1] += 360
    if euler[2] > 180: euler[2] -= 360
    elif euler[2] < -180: euler[2] += 360
    return euler


def euler_to_rotation_matrix(headpose):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
    # Change coordinates system
    euler = np.array([-(headpose[0]-90), -headpose[1], -(headpose[2]+90)])
    # Convert to radians
    rad = euler*(np.pi/180.0)
    cy = np.cos(rad[0])
    sy = np.sin(rad[0])
    cp = np.cos(rad[1])
    sp = np.sin(rad[1])
    cr = np.cos(rad[2])
    sr = np.sin(rad[2])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
    Rp = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
    Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
    return np.matmul(np.matmul(Ry, Rp), Rr)
