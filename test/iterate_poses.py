from pathlib import Path
import numpy as np

path = Path("/Users/ilhamiulugturkkan/Desktop/DL_Project/cars_train/1a1dcd236a1e6133860800e6696b8284")


def txt_to_matrix(text):
    matrix = np.loadtxt(text)          # shape (4, 4)
    matrix = matrix.reshape(4,4)
    return matrix

def get_cam_unit_vector(pose_matrix): # input is pose txt's [4,4] matrix (transformed)
    t = pose_matrix[:3, 3]
    return t / np.linalg.norm(t)  

poses = {}
for txt in sorted((path / "pose").glob("*.txt")):
    pose = txt_to_matrix(txt)
    R = pose[:3, :3]                # rotation
    t = pose[:3, 3]                 # camera position (c1, c2, c3)
    poses[txt.stem] = pose
    print(txt.name, t, get_cam_unit_vector(pose))