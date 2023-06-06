'''
    Input rotation trained, compared FK results
'''

import argparse
import pickle
import torch
import numpy as np
from utils.geometry import mean_angle_error_per_joint_pavllo
from scipy.spatial.transform import Rotation as R
from utils.quaternion import qeuler, euler_to_quaternion
ROOFING_10_JOINTS = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12]

HINGE_JOINTS = [2, 4, 7, 9]
BALL_JOINTS = [1, 3, 6, 8]
SKIP_JOINTS = [0, 5]

JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee',
    'LHip', 'LKnee',
    'Back',
    'RShoulder', 'RElbow',
    'LShoulder', 'LElbow'
]

parser = argparse.ArgumentParser(description='Save predictions for easier analysis pipeline')
parser.add_argument('--records', '-r', required=True, type=str, help='prediction records file')
args = parser.parse_args()

with open(args.records, 'rb') as infile:
    data = pickle.load(infile)
angles_gt = data['rotations_gt']
angles_pred = data['rotations_pred']

# zero out non-moving joint angles
for idx in range(angles_pred.shape[1]):
    if idx not in ROOFING_10_JOINTS:
        angles_pred[:, idx] = np.zeros((3, 3))
n_joints = angles_pred.shape[1]
preds = np.reshape(angles_pred, (-1, 3, 3))
gt = np.reshape(angles_gt, (-1, 3, 3))

quat_preds_14 = torch.from_numpy(R.from_matrix(preds).as_quat()[:, [3, 0, 1, 2]]).float().view(-1, n_joints, 4)
quat_gt_14 = torch.from_numpy(R.from_matrix(gt).as_quat()[:, [3, 0, 1, 2]]).float().view(-1, n_joints, 4)

# qeuler returns arrays in order x, y, z 
predicted_euler = qeuler(quat_preds_14, order='zxy', epsilon=1e-6).numpy()[:, ROOFING_10_JOINTS] # F, J, 3
expected_euler = qeuler(quat_gt_14, order='zxy', epsilon=1e-6).numpy()[:, ROOFING_10_JOINTS] # F, J, 3

# clean up some noisy axes (i.e. xy for hinge joints)
for idx in range(predicted_euler.shape[1]):
    if idx in HINGE_JOINTS:
        predicted_euler[:, idx, 0:2] = np.zeros((2,))
        expected_euler[:, idx, 0:2] = np.zeros((2,))

quat_preds = torch.from_numpy(euler_to_quaternion(predicted_euler, 'zxy')).view(-1, 10, 4)
quat_gt = torch.from_numpy(euler_to_quaternion(expected_euler, 'zxy')).view(-1, 10, 4)

n_joints = quat_preds.shape[1]

per_joint_rotation_errors = mean_angle_error_per_joint_pavllo(quat_preds, quat_gt, n_joints=n_joints)
mean_error = []
for j, e in zip(JOINT_NAMES, per_joint_rotation_errors):
    if JOINT_NAMES.index(j) in SKIP_JOINTS:
        continue
    if JOINT_NAMES.index(j) in HINGE_JOINTS:
        mean_error.append(e * 3)
        print(f'{j}: {np.rad2deg(e * 3):.2f} deg - {(e * 3):.2f} rad')
    else:
        mean_error.append(e)
        print(f'{j}: {np.rad2deg(e):.2f} deg - {e:.2f} rad')

print(f'Average: {np.rad2deg(sum(mean_error)/len(mean_error)):.2f} deg - {sum(mean_error)/len(mean_error):.2f} rad')