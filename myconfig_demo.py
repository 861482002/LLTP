# -*- codeing = utf-8 -*-
# @Time : 2023/7/4 16:19
# @Author : 张庭恺
# @File : myconfig_demo.py
# @Software : PyCharm
_base_ = [
    # 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_test.py',
    'checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'

]
data_root = 'D:/A_github/mmdetection3d/data/kitti_tiny'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    persistent_workers=True,
    train = dict(
        dataset = dict(
            ann_file = 'D:/A_github/mmdetection3d/data/kitti/kitti_infos_val.pkl',
            data_root = 'D:/A_github/mmdetection3d/data/kitti',
            # ann_file = 'D:/A_github/mmdetection3d/data/kitti_tiny/kitti_infos_train.pkl',
            # pts_prefix = 'D:/A_github/mmdetection3d/data/kitti_tiny/training/velodyne_reduced'
            pts_prefix = 'D:/A_github/mmdetection3d/data/kitti/training/velodyne_reduced_8c_pointrend_random'
        )
    ),
    val = dict(
        ann_file = 'D:/A_github/mmdetection3d/data/kitti_tiny/kitti_infos_val.pkl',
        pts_prefix = 'D:/A_github/mmdetection3d/data/kitti_tiny/training/velodyne_reduced'
        # pts_prefix = 'D:/A_github/mmdetection3d/data/kitti/training/velodyne_reduced_8c_pointrend_random'

    ),
    test = dict(
        type='KittiDataset',
        data_root='D:/A_github/mmdetection3d/data/kitti',
        ann_file='D:/A_github/mmdetection3d/data/kitti_tiny/kitti_infos_test.pkl',
        # split='training',
        split='testing_tiny',
        pts_prefix='velodyne_reduced_8c_pointrend_random',

        )
    )

optimizer = dict(
    type = 'AdamW',lr = 0.001,betas=(0.95,0.99),weight_decay = 0.01
    )

lr_config = None
momentun_config = None


runner = dict(max_epochs=1)
checkpoint_config = dict(interval =1)
evaluation = dict(interval = 1)
log_config = dict(interval = 10)

# load_from = '../work_dirs_pointrend_pointpillars_random_8demo/latest.pth'


