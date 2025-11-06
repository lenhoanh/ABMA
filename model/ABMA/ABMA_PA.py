from model.ABMA.ABMA import ABMA

from dataset.process import get_data_sf, get_data_kf, get_data_rf
from dataset.process import get_data_noise, get_data_patch
from dataset.process import get_data_occlusion, get_data_illumination

from dataset.aug import patch_SmoothMixS, apply_gaussian_noise
from dataset.aug import apply_object_occlusion, apply_illumination_change

import torch
import numpy as np
import torch.multiprocessing
from tqdm import tqdm
from model.util.plot import print_screen, AverageMeter


class ABMA_PA(ABMA):
    def __init__(self, cfg, device, logger, run, output_dir):
        super().__init__(cfg, device, logger, run, output_dir)
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='__init__()')
        # ===================================================
        if self.cfg.SYSTEM.phase == 'train':
            self.bz = self.cfg.DATASET.train.bz
            self.loss_abnormal_type = self.cfg.TRAIN.params.loss_abnormal_type
            self.max_prob = 0.05
            self.data_pseudo = None
            self.iter_pseudo = None
            self.pseudo_probs = self.get_pseudo_probs()
            self.sf_list = self.get_sf_list()
            self.setup_aug_data()

    def save_model(self, model_filepath):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='save_model()')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='load_model()')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='save_model_state_dict()')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='load_model_state_dict()')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='save_checkpoint()')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath, is_load_scheduler=True):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='load_checkpoint()')
        return super().load_checkpoint(checkpoint_filepath, is_load_scheduler)

    def transfer_scheduler_checkpoint(self, checkpoint_filepath_source, checkpoint_filepath_destination):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='transfer_scheduler_checkpoint()')
        return super().transfer_scheduler_checkpoint(checkpoint_filepath_source, checkpoint_filepath_destination)

    def train(self):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='train()')
        return super().train()

    def setup_aug_data(self):
        """
        Setup data_pseudo for multiple types based on self.cfg.DATASET.pseudo_types.
        Gọi get_data_*() đã wrap InfiniteDataLoader (num_workers=0) để fix open files & leak.
        No iter_pseudo: Handle iter in _apply_single_type.
        """
        from itertools import cycle  # Import here if needed

        if hasattr(self.cfg.DATASET, 'pseudo_types') and self.cfg.DATASET.pseudo_types:
            self.data_pseudo = {}
            self.iter_pseudo = {}
            for pseudo_type in self.cfg.DATASET.pseudo_types:
                data_dict = None
                if pseudo_type == 'SF':
                    data_dict = get_data_sf(self.cfg)
                elif pseudo_type == 'KF':
                    data_dict = get_data_kf(self.cfg)
                elif pseudo_type == 'RF':
                    data_dict = get_data_rf(self.cfg)
                elif pseudo_type == 'Patch':
                    data_dict = get_data_patch(self.cfg)
                elif pseudo_type == 'Noise':
                    data_dict = get_data_noise(self.cfg)
                elif pseudo_type == 'Occlusion':
                    data_dict = get_data_occlusion(self.cfg)
                elif pseudo_type == 'Illumination':
                    data_dict = get_data_illumination(self.cfg)
                else:
                    data_dict = {}  # Skip invalid type

                self.data_pseudo[pseudo_type] = data_dict
        else:
            # Fallback
            self.data_pseudo = get_data_sf(self.cfg)

    def get_pseudo_probs(self):
        """
        Return dict {pseudo_type: p} cho tất cả pseudo_types.
        """
        probs = {}
        if hasattr(self.cfg.DATASET, 'pseudo_types') and self.cfg.DATASET.pseudo_types:
            for pseudo_type in self.cfg.DATASET.pseudo_types:
                dict_pseudo = getattr(self.cfg.DATASET, f'dict_{pseudo_type.lower()}', {})
                probs[pseudo_type] = dict_pseudo.get('p', 0.001)
        else:
            # Fallback
            probs['SF'] = self.cfg.DATASET.dict_sf.p if hasattr(self.cfg.DATASET, 'dict_sf') else 0.001
        return probs

    def get_sf_list(self):
        sf_list = [2]
        if hasattr(self.cfg.DATASET, 'pseudo_types') and 'SF' in self.cfg.DATASET.pseudo_types:
            sf_list = self.cfg.DATASET.dict_sf.s
        return sf_list

    def get_clip_abnormal(self, clip_normal=None):
        """
        Return None nếu normal, hoặc {'clip': clip_abnormal, 'type': selected_type} nếu apply pseudo.
        Hỗ trợ riêng lẻ/kết hợp qua self.cfg.DATASET.pseudo_types (list str, e.g., ['Noise'], ['Noise', 'Illumination'], ['SF', 'Noise', 'Patch'] cho >2 loại).
        - Per sample: Random chọn 1 loại từ pseudo_types (equal prob), apply nếu trúng p của loại đó.
        - Không sequential; chỉ 1 loại per clip.
        """
        if hasattr(self.cfg.DATASET, 'pseudo_types') and self.cfg.DATASET.pseudo_types:
            pseudo_types = self.cfg.DATASET.pseudo_types
            if len(pseudo_types) > 0:
                # Random chọn 1 loại (equal probability)
                selected_type = np.random.choice(pseudo_types)
                data_dict = self.data_pseudo.get(selected_type, {})
                p = data_dict["p"]
                random_prob = np.random.rand()
                if random_prob < p and clip_normal is not None:
                    clip_abnormal = self._apply_single_type(clip_normal, selected_type)
                    if clip_abnormal is not None:
                        return {'clip': clip_abnormal.to(self.device), 'type': selected_type}
        return None

    def _apply_single_type(self, clip, pseudo_type):
        """
        Apply 1 loại pseudo (thống nhất tất cả, bao gồm SF/KF/RF).
        Use iter(dataloader) on-the-fly với InfiniteDataLoader (tự reset, no leak).
        """
        data_dict = self.data_pseudo.get(pseudo_type, {})
        dataloader = data_dict.get('dataloader', None)
        clip_abnormal = None

        if pseudo_type in ['SF', 'KF', 'RF']:
            if dataloader is not None:
                iterator = iter(dataloader)
                try:
                    data_pseudo = next(iterator)
                    clip_abnormal = data_pseudo[0]
                except StopIteration:
                    # Manual reset: Shuffle dataset nếu cần, rồi iter lại
                    dataloader.sampler = torch.utils.data.RandomSampler(
                        dataloader.dataset)  # Re-shuffle nếu shuffle=True
                    iterator = iter(dataloader)
                    data_pseudo = next(iterator)
                    clip_abnormal = data_pseudo[0]

            return clip_abnormal
        elif pseudo_type == 'Noise':
            sigma = data_dict['sigma']
            clip_abnormal = apply_gaussian_noise(clip, mean=0.0, std=sigma)
            return clip_abnormal
        elif pseudo_type == 'Patch':
            if dataloader is not None:
                iterator = iter(dataloader)
                try:
                    data_pseudo = next(iterator)
                except StopIteration:
                    # Reset
                    dataloader.sampler = torch.utils.data.RandomSampler(dataloader.dataset)
                    iterator = iter(dataloader)
                    data_pseudo = next(iterator)

                alpha = data_dict.get('alpha', 0.5)
                beta = data_dict.get('beta', 0)
                clip_mask, mask = patch_SmoothMixS(clip.unsqueeze(0),
                                                   data_pseudo[0].squeeze(0),
                                                   max_size=alpha,
                                                   c=1,
                                                   h=self.cfg.DATASET.width,
                                                   w=self.cfg.DATASET.height,
                                                   max_move=beta)
                clip_abnormal = clip_mask.squeeze(0)
            return clip_abnormal
        elif pseudo_type == 'Occlusion':
            max_area_ratio = data_dict.get('max_area_ratio', 0.3)
            clip_abnormal = apply_object_occlusion(clip, max_area_ratio=max_area_ratio)
            return clip_abnormal
        elif pseudo_type == 'Illumination':
            factor_range = data_dict.get('factor_range', (0.5, 1.5))
            clip_abnormal = apply_illumination_change(clip, factor_range=factor_range)
            return clip_abnormal
        return clip_abnormal

    def calculate_loss_inside_bz(self, clip):
        # Khởi tạo dictionary chứa danh sách loss của từng sample
        sample_losses = {key: [] for key in
                         ['total_loss',
                          'loss_intensity_past', 'loss_compact_past', 'loss_separate_past',
                          'loss_intensity_future', 'loss_compact_future', 'loss_separate_future']}

        # Đọc từ config
        pseudo_abnormal_types = self.cfg.DATASET.get('pseudo_abnormal_types',
                                                     ['SF', 'KF', 'RF', 'Noise', 'Patch', 'Occlusion', 'Illumination'])
        pseudo_normal_types = self.cfg.DATASET.get('pseudo_normal_types', ['A', 'B'])

        for b in range(self.bz):
            current_clip = clip[b].unsqueeze(0)  # Lấy tensor của sample hiện tại
            pseudo_result = self.get_clip_abnormal(clip_normal=clip[b])
            is_pseudo = pseudo_result is not None
            pseudo_type = pseudo_result['type'] if is_pseudo else None

            if is_pseudo:
                current_clip = pseudo_result['clip'].unsqueeze(0)

            # Tính loss
            losses, _ = self.calculate_loss_mean(current_clip, is_pseudo)

            # Xử lý loss dựa trên loại sample
            if is_pseudo:
                loss_strong_scale = self.cfg.TRAIN.params.loss_strong_scale
                is_abnormal = pseudo_type in pseudo_abnormal_types
                if is_abnormal:
                    # Flip sign cho strong/severe pseudo-anomalies
                    total_pseudo_loss = -loss_strong_scale * losses['total_loss']
                    if self.loss_abnormal_type == 'I':
                        total_pseudo_loss = -loss_strong_scale * (
                                losses['loss_intensity_past'] + losses['loss_intensity_future'])
                    elif self.loss_abnormal_type == 'IC':
                        total_pseudo_loss = -loss_strong_scale * (
                                    losses['loss_intensity_past'] + losses['loss_compact_past']
                                    + losses['loss_intensity_future'] + losses[
                                        'loss_compact_future'])
                    else:  # 'ICS'
                        total_pseudo_loss = -loss_strong_scale * losses['total_loss']
                    pseudo_loss = {'total_loss': total_pseudo_loss,
                                   'loss_intensity_past': -losses['loss_intensity_past'],
                                   'loss_compact_past': -losses['loss_compact_past'],
                                   'loss_separate_past': torch.tensor(0.0, device=self.device),
                                   'loss_intensity_future': -losses['loss_intensity_future'],
                                   'loss_compact_future': -losses['loss_compact_future'],
                                   'loss_separate_future': torch.tensor(0.0, device=self.device)
                                   }
                else:
                    # Không flip cho subtle, weight thấp (λ=0.3)
                    loss_subtle_scale = self.cfg.TRAIN.params.loss_subtle_scale  # 0.3
                    total_pseudo_loss = loss_subtle_scale * losses['total_loss']
                    pseudo_loss = {'total_loss': total_pseudo_loss,
                                   'loss_intensity_past': loss_subtle_scale * losses['loss_intensity_past'],
                                   'loss_compact_past': loss_subtle_scale * losses['loss_compact_past'],
                                   'loss_separate_past': loss_subtle_scale * losses['loss_separate_past'],
                                   # Giữ separate cho subtle
                                   'loss_intensity_future': loss_subtle_scale * losses['loss_intensity_future'],
                                   'loss_compact_future': loss_subtle_scale * losses['loss_compact_future'],
                                   'loss_separate_future': loss_subtle_scale * losses['loss_separate_future']
                                   }

                sample_loss = pseudo_loss
            else:  # Normal sample
                sample_loss = losses

            # Cập nhật từng loss của sample vào dictionary
            for key in sample_losses.keys():
                sample_losses[key].append(sample_loss[key])

        # Tính trung bình các loss qua batch (sử dụng torch.mean trên tensor stack)
        avg_losses = {
            'total_loss': torch.mean(torch.stack(sample_losses['total_loss'])),
            'loss_intensity_past': torch.mean(torch.stack(sample_losses['loss_intensity_past'])),
            'loss_compact_past': torch.mean(torch.stack(sample_losses['loss_compact_past'])),
            'loss_separate_past': torch.mean(torch.stack(sample_losses['loss_separate_past'])),
            'loss_intensity_future': torch.mean(torch.stack(sample_losses['loss_intensity_future'])),
            'loss_compact_future': torch.mean(torch.stack(sample_losses['loss_compact_future'])),
            'loss_separate_future': torch.mean(torch.stack(sample_losses['loss_separate_future'])),
        }

        return avg_losses

    def train_epoch(self, epoch):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='train_epoch()')
        self.epoch = epoch + 1

        print_screen(logger=self.logger, key='batch_size', val=self.bz)
        print_screen(logger=self.logger, key='loss_abnormal_type', val=self.loss_abnormal_type)

        model_loss_type = getattr(self.cfg.MODEL.params, "loss_type", None)  # Nếu không tồn tại, trả về None
        print_screen(logger=self.logger, key='cfg.MODEL.params.loss_type', val=model_loss_type)

        if 'SF' in self.cfg.MODEL.method:
            print_screen(logger=self.logger, key='skip_frames', val=self.cfg.DATASET.dict_sf.s)

        # Danh sách các key loss cần theo dõi (có thể bổ sung thêm nếu cần)
        loss_keys = ['total_loss',
                     'loss_intensity_past', 'loss_compact_past', 'loss_separate_past',
                     'loss_intensity_future', 'loss_compact_future', 'loss_separate_future']
        # Khởi tạo AverageMeter cho mỗi key
        train_loss_dict = {key: AverageMeter() for key in loss_keys}

        # self.abnormal_accumulated = False
        for idx, clip in enumerate(tqdm(self.train_dataloader, desc="Processing Clips")):
            # clip.shape = torch.Size([4, 15, 256, 256]);  clip.device='cpu'
            clip = clip.to(self.device)
            avg_losses = self.calculate_loss_inside_bz(clip)
            total_loss = avg_losses['total_loss']
            self.optimizer.zero_grad()
            total_loss.backward()

            self.optimizer.step()

            batch_size = clip.size(0)
            # Cập nhật các loss vào AverageMeter qua dictionary
            for key in loss_keys:
                train_loss_dict[key].update(avg_losses[key].item(), batch_size)

            # Explicitly delete tensors and trigger garbage collection
            del clip, total_loss, avg_losses

        for key in loss_keys:
            print_screen(logger=self.logger, key=key, val=(train_loss_dict[key].avg,), format_string='{:.9f}')

        # average the loss_epoch
        self.loss_epoch = train_loss_dict['total_loss'].avg
        return self.loss_epoch

    def test(self):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='test()')
        return super().test()

    def evaluate(self):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='evaluate()')
        return super().evaluate()
