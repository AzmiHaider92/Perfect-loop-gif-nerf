import json
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial.transform import Rotation as R


def base_cam_points(json_path):
    points = []
    with open(json_path, 'r') as f:
        data = json.load(f)
        frames = data['path']
        for i, frame in enumerate(frames):
            # frame_num = frame['file_path']
            p = frame["T"]
            points.append(p + [i])
    return np.array(points)

def points_from_transforms(json_path):
    points = []
    camera_angle_x, camera_angle_y, cx, cy, w, h = 0,0,0,0,0,0
    with open(json_path, 'r') as f:
        data = json.load(f)
        frames = data['frames']
        try:
            camera_angle_x, camera_angle_y, cx, cy, w, h = \
                data['camera_angle_x'],  data['camera_angle_y'], data['cx'], data['cy'], data['w'], data['h']
        except:
            print("nonono")
        for i, frame in enumerate(frames):
            #frame_num = frame['file_path']
            transform = np.array(frame['transform_matrix'])
            xyz = transform[:3, 3]
            points.append(list(xyz) + [i])

    points = np.array(points)
    return points, camera_angle_x, camera_angle_y, cx, cy, w, h


def near_far(points):
    radius = np.max(np.sqrt(np.sum(points[:,:2] ** 2, axis=1)))
    z = np.mean(points[:,2])


    return radius, z


def show_figure(points, title='', c='b'):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(points.shape[0]):  # plot each point + it's index as text above
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], color=c)
        ax.text(points[i, 0], points[i, 1], points[i, 2],
                '%s' % (str(points[i, 3])), size=5, zorder=1,
                color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)


def rotate(x, y, r):
    rx = (x*math.cos(r)) - (y*math.sin(r))
    ry = (y*math.cos(r)) + (x*math.sin(r))
    return (rx, ry)


def points_on_circle(radius, center, num_points=360):
    arc = (2 * math.pi) / num_points  # what is the angle between two of the points
    points = []
    for p in range(num_points):
        (px, py) = rotate(0, radius, arc * p)
        px += center[0]
        py += center[1]
        points.append((px, py))
    return np.array(points)


def points_to_transforms(points):
    transforms = []
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    for p in points:
        t = np.array([p,1])
        t = t.reshape([3, 1])
        theta , yaw, roll = 0, 0 ,0
        R = 0
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)




if __name__ == '__main__':
    json_path_colmap = r'C:\Users\azmih\Desktop\Projects\ComputerVisionLab\TensoRF\data\scene1\transforms_train.json'
    #json_path_colmap = r'C:\Users\azmih\Desktop\Projects\Instant-NGP-for-RTX-3000-and-4000\data\nerf\62_4316_10771\base_cam.json'

    cm_camera_points, camera_angle_x, camera_angle_y, cx, cy, w, h = points_from_transforms(json_path_colmap)
    show_figure(cm_camera_points, 'scene1_gen')
    #cm_camera_points = base_cam_points(json_path_colmap)
    arr = np.load(r"C:\Users\azmih\Desktop\Projects\ComputerVisionLab\TensoRF\log\tensorf_1_VM\c2w.npy")
    points = []
    for i, frame in enumerate(arr):
        # frame_num = frame['file_path']
        xyz = frame[:3, 3]
        points.append(list(xyz) + [i])
    cm_camera_points = np.array(points)
    show_figure(cm_camera_points, 'scene1')
    r, z = near_far(cm_camera_points)

    # new camera positions
    '''
    tdpoints = points_on_circle(r, (0,0))
    #new_transforms = points_to_transforms(tdpoints)
    o = np.ones((tdpoints.shape[0], 1))
    new_camera_points = np.hstack([tdpoints, z*o, o])
    show_figure(new_camera_points, 'scene1_gen')
    '''

    plt.show()
    end=1