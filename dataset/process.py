from .video import VideoDataset, VideoDataset_sf, VideoDataset_rf, VideoDataset_kf, VideoDataset_sf_and_normal
from .util import get_transform
from .data_types import dict_clip, dict_patch_ST
from .data_types import dict_sf, dict_rf, dict_patch, dict_kf, dict_obj
from .data_types import dict_fusion, dict_noise, dict_shake, dict_occlusion, dict_illumination
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import os
import torch.utils.data as data


def get_train_dataloader(cfg):
    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.train.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':  # MNAD
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":  # MemAE
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    dataset = VideoDataset(data_clip, dataset_path)  # a clip = 5 frames (256,256,3)
    dataloader = data.DataLoader(
        dataset,
        batch_size=cfg.DATASET.train.bz,  # 4
        shuffle=cfg.DATASET.train.shuffle,  # cfg.DATASET.train.shuffle
        num_workers=cfg.DATASET.train.num_workers,  # cfg.DATASET.train.num_workers,  # 4 or 8; 4 * num_gpus_available,
        pin_memory=False,  # False (to reduce RAM consumption): gpustat (VRAM), htop
        drop_last=cfg.DATASET.train.drop_last
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TRAIN dataset: {num_samples}")

    return dataloader


def get_data_sf(cfg):
    """
    get data_sf: based on dict_sf (p, s, dataloader)
    """
    data_sf = dict(dict_sf)
    data_sf['p'] = cfg.DATASET.dict_sf.p
    data_sf['s'] = cfg.DATASET.dict_sf.s
    data_sf['dataset'] = None
    data_sf['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.train.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_sf.p > 0:
        dataset = VideoDataset_sf(data_clip, dataset_path, data_sf['s'])
        data_sf['dataset'] = dataset
        '''
        data_sf['dataloader'] = data.DataLoader(
            dataset,
            batch_size=cfg.DATASET.train.bz,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=cfg.DATASET.train.num_workers,
            pin_memory=False,
            drop_last=True
        )
        '''
        # Wrap dataloader với num_workers=0 (fix open files), pin_memory=False, persistent_workers=False
        data_sf['dataloader'] = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,  # Key fix: Không parallel để tránh "open files"
            pin_memory=False,  # Giảm memory pressure
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset SKIP_FRAMES: {num_samples}")
    return data_sf


def get_data_sf_and_normal(cfg):
    data_sf_and_normal = dict(dict_sf)
    data_sf_and_normal['p'] = cfg.DATASET.dict_sf.p
    data_sf_and_normal['s'] = cfg.DATASET.dict_sf.s
    data_sf_and_normal['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.train.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_sf.p > 0:
        dataset = VideoDataset_sf_and_normal(data_clip, dataset_path, data_sf_and_normal['s'])
        data_sf_and_normal['dataloader'] = data.DataLoader(
            dataset,
            batch_size=cfg.DATASET.train.bz,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=cfg.DATASET.train.num_workers,
            pin_memory=False,
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset SKIP_FRAMES: {num_samples}")
    return data_sf_and_normal


def get_data_kf(cfg):
    data_kf = dict(dict_kf)
    data_kf['p'] = cfg.DATASET.dict_kf.p
    data_kf['dataset'] = None
    data_kf['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.keyframes_dir)
    print('dataset_path =', dataset_path)
    if not os.path.exists(dataset_path):
        print(f'{dataset_path} is NOT exists !!!')
        return data_kf
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.train.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_kf.p > 0:
        dataset = VideoDataset_kf(data_clip, dataset_path)  # VideoDataset(data_clip, dataset_path)  #
        data_kf['dataset'] = dataset
        '''
        data_kf['dataloader'] = data.DataLoader(
            dataset,
            batch_size=cfg.DATASET.train.bz,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=cfg.DATASET.train.num_workers,
            pin_memory=False,
            drop_last=True
        )
        '''

        # Wrap dataloader với num_workers=0 (fix open files), pin_memory=False, persistent_workers=False
        data_kf['dataloader'] = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,  # Key fix: Không parallel để tránh "open files"
            pin_memory=False,  # Giảm memory pressure
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset KEY_FRAMES: {num_samples}")

    return data_kf


def get_data_obj(cfg):
    data_obj = dict(dict_obj)
    data_obj['p'] = cfg.DATASET.dict_obj.p
    data_obj['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.obj_dir)
    print('dataset_path =', dataset_path)
    if not os.path.exists(dataset_path):
        print(f'{dataset_path} is NOT exists !!!')
        return data_obj
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.train.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_obj.p > 0:
        dataset = VideoDataset(data_clip, dataset_path)  # VideoDataset(data_clip, dataset_path)  #
        data_obj['dataloader'] = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset OBJECT: {num_samples}")

    return data_obj


def get_data_rf(cfg):
    data_rf = dict(dict_rf)
    data_rf['p'] = cfg.DATASET.dict_rf.p
    data_rf['r'] = cfg.DATASET.dict_rf.r
    data_rf['dataset'] = None
    data_rf['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.train.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_rf.p > 0:
        dataset = VideoDataset_rf(data_clip, dataset_path, data_rf['r'])
        data_rf['dataset'] = dataset
        '''
        data_rf['dataloader'] = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        '''

        # Wrap dataloader với num_workers=0 (fix open files), pin_memory=False, persistent_workers=False
        data_rf['dataloader'] = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,  # Key fix: Không parallel để tránh "open files"
            pin_memory=False,  # Giảm memory pressure
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset REPEAT: {num_samples}")

    return data_rf


def get_data_patch(cfg):
    data_patch = dict(dict_patch)
    data_patch['p'] = cfg.DATASET.dict_patch.p
    data_patch['alpha'] = cfg.DATASET.dict_patch.alpha
    data_patch['beta'] = cfg.DATASET.dict_patch.beta
    data_patch['technique'] = cfg.DATASET.dict_patch.technique
    data_patch['intruder'] = cfg.DATASET.dict_patch.intruder
    data_patch['dataset'] = None
    data_patch['dataloader'] = None

    if cfg.DATASET.dict_patch.p > 0 and data_patch['intruder'] == 'CIFAR-100':
        path = os.path.join(os.getcwd(), 'dataset', 'CIFAR100')
        # dataset.__getitem__() return (image, target) where target is index of the target class.
        dataset = CIFAR100(path,
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())
        data_patch['dataset'] = dataset
        '''
        cifar100_dataloader = data.DataLoader(dataset,
                                                          batch_size=1,
                                                          shuffle=cfg.DATASET.train.shuffle,
                                                          num_workers=0,
                                                          pin_memory=False,
                                                          drop_last=True)
        '''
        # Wrap dataloader với num_workers=0 (fix open files), pin_memory=False, persistent_workers=False
        cifar100_dataloader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,  # Key fix: Không parallel để tránh "open files"
            pin_memory=False,  # Giảm memory pressure
            drop_last=True
        )
        data_patch['dataloader'] = cifar100_dataloader
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset CIFAR100: {num_samples}")

    return data_patch


def get_data_fusion(cfg):
    data_fusion = dict(dict_fusion)
    data_fusion['p'] = cfg.DATASET.dict_fusion.p
    return data_fusion


def get_data_noise(cfg):
    data_noise = dict(dict_noise)
    data_noise['p'] = cfg.DATASET.dict_noise.p
    data_noise['sigma'] = cfg.DATASET.dict_noise.sigma
    return data_noise


def get_data_occlusion(cfg):
    data_occlusion = dict(dict_occlusion)
    data_occlusion['p'] = cfg.DATASET.dict_occlusion.p
    data_occlusion['max_area_ratio'] = cfg.DATASET.dict_occlusion.max_area_ratio
    return data_occlusion


def get_data_illumination(cfg):
    data_illumination = dict(dict_illumination)
    data_illumination['p'] = cfg.DATASET.dict_illumination.p
    data_illumination['factor_range'] = cfg.DATASET.dict_illumination.factor_range
    return data_illumination

def get_data_shake(cfg):
    data_shake = dict(dict_shake)
    data_shake['p'] = cfg.DATASET.dict_shake.p
    data_shake['rot'] = cfg.DATASET.dict_shake.rot
    data_shake['trans'] = cfg.DATASET.dict_shake.trans
    return data_shake


# ============================================================
# ============================================================
def get_test_dataset(cfg):
    """
    param: sample_type = 'video' (MNAD_Pred, MNAD_Pred, ASTNet)
            sample_type = clips_by_video' (FastAno, STEAL, PAMAE4)
    """
    # fullpath of test_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.test.test_set)
    print('dataset_path = ', dataset_path)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.test.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.test.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])
    dataset = VideoDataset(data_clip, dataset_path)
    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TEST dataset: {num_samples}")
    return dataloader


# ============================================================
# ============================================================
def get_dataset_to_generate_OpticalFlow(cfg, dataset_path):
    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.train.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])
    dataset = VideoDataset(data_clip, dataset_path)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return dataset, dataloader
