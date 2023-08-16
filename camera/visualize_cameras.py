import os
import numpy as np
import json
import matplotlib.pyplot as plt

coord_trans = np.diag([1, -1, -1, 1])


def read_transforms(transforms_path):
    poses = []
    T = []
    if os.path.isdir(transforms_path):
        pose_files = os.listdir(transforms_path)
        for pose_file in pose_files:
            c2w = np.loadtxt(os.path.join(transforms_path, pose_file),
                                      dtype=np.float32).reshape(4, 4)
            #c2w = c2w @ coord_trans
            T.append(c2w[:-1, -1])
            poses.append(c2w)
    else: # json file
        with open(transforms_path, 'r') as f:
            data = json.load(f)
            frames = data['frames']
            for frame in frames:
                c2w = np.array(frame['transform_matrix'])
                c2w = c2w @ coord_trans
                T.append(c2w[:-1, -1])
                poses.append(c2w)

    return poses, T

from TensoRF.camera.camera_visualizer import CameraPoseVisualizer
def visualize_transforms(Transforms, cameraPositions):
    cameraPositions = np.array(cameraPositions)
    m=5
    Transforms = Transforms[:5]
    visualizer = CameraPoseVisualizer([-m+np.min(cameraPositions[:, 0]), m+np.max(cameraPositions[:, 0])],
                                      [-m+np.min(cameraPositions[:, 1]), m+np.max(cameraPositions[:, 1])],
                                      [-m-np.max(np.abs(cameraPositions[:, 2])), m+np.max(np.abs(cameraPositions[:, 2]))])
    max_frame_length = len(Transforms)
    for idx_frame, T in enumerate(Transforms):
        visualizer.extrinsic2pyramid(T, plt.cm.rainbow(idx_frame / max_frame_length), 10)

    visualizer.colorbar(max_frame_length)
    visualizer.show()
    e=1

if __name__ == "__main__":
    transforms_path = r"C:\Users\azmih\Desktop\Projects\TensoRF\data\srn_cars\srn_cars_one\1a1dcd236a1e6133860800e6696b8284\pose"
    #transforms_path = r"C:\Users\azmih\Desktop\Projects\TensoRF\data\co3d\vase_all_10scenes\108_12867_22800\transforms_train.json"
    Transforms, cameraPositions = read_transforms(transforms_path)
    visualize_transforms(Transforms, cameraPositions)