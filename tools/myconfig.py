_base_ = [
    # 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'
    # 'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    # 'checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'
    # 'configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py'
    'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    # 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_8c.py'
    # 'checkpoints/point_rcnn_2x8_kitti-3d-3classes.py'
]
data_root = 'D:/A_github/mmdetection3d/data/kitti'
data = dict(
    samples_per_gpu=2,   #batchsize批量
    workers_per_gpu=1,   #dataloader 的数量 进程数
    persistent_workers=True,
    train = dict(
        dataset = dict(
            # ann_file = 'data/kitti/kitti_infos_val.pkl'
            data_root = 'D:/A_github/mmdetection3d/data/kitti',
            ann_file = 'D:/A_github/mmdetection3d/data/kitti/kitti_infos_train.pkl',
            pts_prefix = 'D:/A_github/mmdetection3d/data/kitti/training/velodyne_reduced'
            # pts_prefix = 'D:/A_github/mmdetection3d/data/kitti/training/velodyne_reduced_8c_deeplab_random'
        )
    ),
    val = dict(
        ann_file = 'D:/A_github/mmdetection3d/data/kitti/kitti_infos_val.pkl',
        pts_prefix = 'D:/A_github/mmdetection3d/data/kitti/training/velodyne_reduced'
        # pts_prefix = 'D:/A_github/mmdetection3d/data/kitti/training/velodyne_reduced_8c_deeplab_random'

    )
    # test = dict(
    #         ann_file='data/kitti/kitti_infos_test.pkl',
    #         split='testing',
    #
    #     )
    )

optimizer = dict(
    type = 'AdamW',lr = 0.001,betas=(0.95,0.99),weight_decay = 0.01
    )

lr_config = None
momentun_config = None
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 0.0001),
#     cyclic_times=1,
#     step_ratio_up=0.4)
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.8947368421052632, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4)

runner = dict(max_epochs=20)
checkpoint_config = dict(interval =2)
evaluation = dict(interval = 2)
log_config = dict(interval = 50)
# load_from = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
# load_from = 'checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20210831_022017-454a5344.pth'
# load_from = 'checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth'
# load_from = 'checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20210831_022017-454a5344.pth'
# load_from = '../work_dirs_pointrend_random_1epoch/latest.pth'
# load_from = '../work_dirs_pointrend_pointpillars_random_8epoch/latest.pth'
# load_from = '../work_dir_pillars_tr_local_15_epoch/latest.pth'
# resume_from = 'work_dirs_pointrend_pointpillars_select/latest.pth'
resume_from = '../work_dir_pillars_mobilenetv3_10_epoch/latest.pth'

