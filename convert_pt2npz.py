import torch
import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation as sRot

def main():
    parser = argparse.ArgumentParser(description='Convert SMPL parameters from .pt to .npz with coordinate transformation')
    parser.add_argument('--input', '-i', required=True, help='Input .pt file path')
    parser.add_argument('--output', '-o', required=True, help='Output .npz file path')
    parser.add_argument('--height', '-H', type=float, default=0.92, help='Initial Z height (default: 0.92m)')
    args = parser.parse_args()

    pt_file = args.input
    output_npz = args.output
    initial_height = args.height

    try:
        # 加载数据
        data = torch.load(pt_file, map_location='cpu', weights_only=True)
        smpl_params_global = data['smpl_params_global']

        np_data = {k: v.cpu().numpy() if torch.is_tensor(v) else v
                   for k, v in smpl_params_global.items()}

        if 'global_orient' in np_data and 'body_pose' in np_data:
            N = np_data['body_pose'].shape[0]
            global_orient = np_data['global_orient'].reshape(N, 3)
            body_pose = np_data['body_pose'].reshape(N, 63)
            zeros6 = np.zeros((N, 6), dtype=body_pose.dtype)
            poses = np.concatenate([global_orient, body_pose, zeros6], axis=1)
            print(f"[INFO] poses constructed: {poses.shape}")
        else:
            raise ValueError("Missing 'global_orient' or 'body_pose' in the input!")
        
        poses = poses[:, :66]
        print(f"[INFO] Final poses shape: {poses.shape}")

        # Translation adjustment
        if 'transl' in np_data:
            np_data['trans'] = np_data.pop('transl')
        trans = np_data['trans']  # (N, 3)

        # ===== 添加坐标系转换 =====
        # 旋转：绕X轴旋转90度（将Z轴调成竖直向上）
        transform = sRot.from_euler('xyz', [np.pi/2, 0, 0], degrees=False)

        # 替换 root joint 的旋转
        pose_aa = poses.copy()  # (N, 66)
        pose_aa[:, :3] = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()

        # 变换 translation 到新坐标系，并修正初始Z高度
        trans_new = trans.dot(transform.as_matrix().T)
        trans_new[:, 2] = trans_new[:, 2] - (trans_new[0, 2] - initial_height)

        # 保存新数据
        np_data['poses'] = pose_aa
        np_data['trans'] = trans_new
        np_data['mocap_framerate'] = 30
        np_data['gender'] = 'neutral'

        # 删除旧字段
        np_data.pop('body_pose', None)
        np_data.pop('global_orient', None)
        np_data['betas'] = np_data['betas'][0]
        print(f"[INFO] Betas shape: {np_data['betas'].shape}")

        np.savez(output_npz, **np_data)
        print(f"[SAVE] Converted + transformed .pt → .npz: {output_npz}")

    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    main()    