import json
import numpy as np
import math
import torch
import matplotlib.pyplot as plt


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


def points_from_transforms(c2w):
    points = []
    for i, transform in enumerate(c2w):
        xyz = transform[:3, 3]
        points.append(list(xyz) + [i])
    points = np.array(points)
    return points


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


def show_figure2D(points, title='', c='b'):
    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(points.shape[0]):  # plot each point + it's index as text above
        ax.scatter(points[i, 0], points[i, 1], color=c)
        ax.text(points[i, 0], points[i, 1],
                '%s' % (str(points[i, 2])), size=10, zorder=1,
                color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
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


def model_f(x,a,b,c):
  return a*x**3 + b*x + c


from scipy.optimize import curve_fit


def link_cam_points(s_transform, e_transform, num=10, curvfit=0):
    T = []

    ts = np.linspace(0.0, 1.0, num=num)
    s = s_transform[0]
    e = e_transform[-1]
    T = [(1-t) * e + t * s for t in ts]
    T = torch.stack(T)
    if curvfit: # 2nd
        transforms = np.vstack((e_transform, s_transform))
        loc = transforms[:,0:2,3]
        popt, pcov = curve_fit(model_f, loc[:,0], loc[:,1], p0=[3, 2, -16])
        a_opt, b_opt, c_opt = popt
        loc2 = T[:,0:2,3]
        y_fitted = model_f(loc2[:,0], a_opt, b_opt, c_opt)
        T[:, 1, 3] = y_fitted

    return T

def fix_path(c2w, margin = 10, num_added_frames = 20 , curvefit=0):
    added_image_indices = np.zeros((len(c2w),1))
    cm_camera_points = points_from_transforms(c2w)
    cm_camera_points2d = np.array([cm_camera_points[:, 0], cm_camera_points[:, 1], cm_camera_points[:, 3]]).T
    show_figure2D(cm_camera_points2d, 'scene1')

    m = np.mean(cm_camera_points2d, axis=0)
    angles = 180 / np.pi * np.arctan2(cm_camera_points2d[:, 1] - m[1], cm_camera_points[:, 0] - m[0])

    sp_angle = angles[0]
    a = np.abs(angles - sp_angle)
    localminimum = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    localminimum_indices = [i for i, x in enumerate(localminimum) if x]
    c = np.max(localminimum_indices)

    # completed a 360. delete every position after 360
    if c < len(c2w)-1:
        c2w = c2w[:c]
        added_image_indices = added_image_indices[:c]

    # remove margin from the end
    c2w_fixed = c2w[:-margin]
    added_image_indices = added_image_indices[:-margin]

    fit_margin = 4
    bridge = link_cam_points(c2w_fixed[0:fit_margin], c2w_fixed[-fit_margin:], num=num_added_frames, curvfit=curvefit)
    bridge_ind = np.ones((len(bridge),1))

    c2w_fixed = torch.cat((c2w_fixed, bridge), 0)
    added_image_indices = np.vstack((added_image_indices, bridge_ind))

    plt.figure()
    plt.plot(a)
    plt.title('distance from starting angle')

    fig, axs = plt.subplots(1, 4)
    axs[0].scatter(cm_camera_points2d[:, 0], cm_camera_points2d[:, 1], color='blue')
    axs[0].scatter(cm_camera_points2d[localminimum_indices, 0], cm_camera_points2d[localminimum_indices, 1],
                       color='red')
    #axs[0].title('overlap point in red')


    axs[1].scatter(cm_camera_points2d[:, 0], cm_camera_points2d[:, 1], color='blue')
    axs[1].scatter(cm_camera_points2d[:margin, 0], cm_camera_points2d[:margin, 1], color='yellow')
    axs[1].scatter(cm_camera_points2d[c-margin:, 0], cm_camera_points2d[c-margin:, 1], color='yellow')
    axs[1].scatter(cm_camera_points2d[c:, 0], cm_camera_points2d[c:, 1], color='red')

    axs[2].scatter(cm_camera_points2d[margin:c-margin, 0], cm_camera_points2d[margin:c-margin, 1], color='blue')

    new_cm_camera_points = points_from_transforms(c2w_fixed)
    new_cm_camera_points = new_cm_camera_points[:,[0,1,3]]
    show_figure2D(new_cm_camera_points, 'scene2')
    axs[3].scatter(new_cm_camera_points[:, 0], new_cm_camera_points[:, 1], color='green')
    #axs[3].scatter(cm_camera_points2d[margin:c - margin, 0], cm_camera_points2d[margin:c - margin, 1], color='blue')

    plt.show()

    return c2w_fixed, added_image_indices


'''
if __name__ == '__main__':
    
    a_points = cm_camera_points2d[:, :2]
    x = a_points[:, 0]
    y = a_points[:, 1]
    ell = EllipseModel()
    ell.estimate(a_points)

    xc, yc, a, b, theta = ell.params

    print("center = ", (xc, yc))
    print("angle of rotation = ", theta)
    print("axes = ", (a, b))

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs[0].scatter(x, y)

    axs[1].scatter(x, y)
    axs[1].scatter(xc, yc, color='red', s=100)
    #axs[1].set_xlim(x.min(), x.max())
    #axs[1].set_ylim(y.min(), y.max())

    ell_patch = Ellipse((xc, yc), 2 * a, 2 * b, theta * 180 / np.pi, edgecolor='red', facecolor='none')

    axs[1].add_patch(ell_patch)

    ##########################
    # evenly distributed points on ellipse
    n = 200
    t = np.random.rand(n) * 2 * np.pi
    p = np.array([a * np.cos(t), b * np.sin(t)]).T

    axs[2].scatter(p[:, 0], p[:, 1])


    ### find closest

    #cm_camera_points = base_cam_points(json_path_colmap)
    arr = np.load(r"C:\\Users\\azmih\\Desktop\\Projects\\ComputerVisionLab\\TensoRF\log\\tensorf_1_VM\c2w.npy")
    points = []
    for i, frame in enumerate(arr):
        # frame_num = frame['file_path']
        xyz = frame[:3, 3]
        points.append(list(xyz) + [i])
    cm_camera_points = np.array(points)
    show_figure(cm_camera_points, 'scene1', c='g')
    r, z = near_far(cm_camera_points)

    # new camera positions

    tdpoints = points_on_circle(r, (0,0))
    #new_transforms = points_to_transforms(tdpoints)
    o = np.ones((tdpoints.shape[0], 1))
    new_camera_points = np.hstack([tdpoints, z*o, o])
    show_figure(new_camera_points, 'scene1_gen')

    plt.show()
    end=1

'''
