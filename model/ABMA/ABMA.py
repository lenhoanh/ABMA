from model.UVAD_Model import UVAD_Model

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import make_dir_path
from model.util.metrics import psnr_1
from model.util.metrics import cal_auc_psnr_feas, calculate_eer, calculate_accuracy, calculate_f1_score
from model.util.auc import plot_auc_manual

from model.util.plot import plot_anomaly_scores_MNAD, plot_anomaly_heatmaps_lines_pastfuture
from model.util.metrics import get_anomaly_rectanges, calculate_anomaly_scores_MNAD
from model.util.plot import save_alpha_auc_values, save_metrics

from dataset.util import is_exists_file
from model.util.loss import MultiLosses_ABMA

import torch
import torch.nn.functional as F
import os
import torch.multiprocessing
from tqdm import tqdm
from model.util.plot import print_screen, MetricsResult, AverageMeter


class ABMA(UVAD_Model):
    def __init__(self, cfg, device, logger, run, output_dir):
        super().__init__(cfg, device, logger, run, output_dir)
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='__init__()')
        # ===================================================
        self.model_type = getattr(cfg.MODEL.params, "convAE", None)
        print_screen(logger=self.logger, key='cfg.MODEL.params.convAE', val=self.model_type)

        print(f'cfg.DATASET.num_frames = {cfg.DATASET.num_frames}')
        num_frames = cfg.DATASET.num_frames - 2
        num_channels = cfg.DATASET.channel
        if self.model_type == 'convAE_abma_k7':  # ABMA, ABMA_SF(avenue, shanghaitech, iitb)
            from .convAE_abma import convAE
            self.model = convAE(num_channels, num_frames, kernel_size=7).to(self.device)
        elif self.model_type == 'convAE_abma_k3':  # ABMA, ABMA_SF (ped2)
            from .convAE_abma import convAE
            self.model = convAE(num_channels, num_frames, kernel_size=3).to(self.device)

        # -------------------------------------------------
        memory_size = cfg.MODEL.params.memory_size
        memory_dim = cfg.MODEL.params.memory_dim
        print(f'cfg.MODEL.params.memory_size = {memory_size}')
        keys_past = F.normalize(torch.rand((memory_size, memory_dim), dtype=torch.float), dim=1)
        keys_future = F.normalize(torch.rand((memory_size, memory_dim), dtype=torch.float), dim=1)
        self.keys = {"past": keys_past, "future": keys_future}
        # ===================================================
        self.loss_fn = MultiLosses_ABMA()
        if self.model is not None:
            self.optimizer = get_optimizer(self.cfg, self.model)
            self.scheduler = get_scheduler(self.cfg, self.optimizer)
        self.psnr_types = cfg.TEST.psnr_types
        self.bz = cfg.DATASET.train.bz
        # ===================================================

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
        try:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.loss_epoch,
                'keys': self.keys,
            }, checkpoint_filepath)
            msg = f'Success: save_checkpoint: {checkpoint_filepath}\n'
            print_screen(logger=self.logger, key=self.cfg.MODEL.method, val=msg)
            return self.epoch
        except FileNotFoundError:
            msg = f"Error: Checkpoint file not found at {checkpoint_filepath}"
            print_screen(logger=self.logger, key=self.cfg.MODEL.method, val=msg)
            return -1
        except Exception as e:
            msg = f'Error: save_checkpoint: {checkpoint_filepath} !!!\n'
            print_screen(logger=self.logger, key=self.cfg.MODEL.method, val=msg)
            msg = f"Error reason: {str(e)}"
            print_screen(logger=self.logger, key=self.cfg.MODEL.method, val=msg)
            return -1

    def load_checkpoint(self, checkpoint_filepath, is_load_scheduler=True):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='load_checkpoint()')
        try:
            is_exists_file(checkpoint_filepath)
            checkpoint = torch.load(checkpoint_filepath, map_location=self.cfg.SYSTEM.device)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # check if optimizer_state_dict exists or not
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
            # Try moving optimizer state to the GPU memory manually after loading it from the checkpoint.
            # torch.is_tensor(v): check if a variable is a PyTorch tensor and not any of its subclasses
            # isinstance(v, torch.Tensor):  If include subclass instances as well,
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            # check if scheduler_state_dict exists or not
            if 'scheduler_state_dict' in checkpoint:
                print("=======>>> scheduler_state_dict EXIST in checkpoint!!!")
            else:
                print("=======>>> scheduler_state_dict NOT_EXIST in checkpoint!!!")

            if is_load_scheduler:
                print("=======>>> load_scheduler")
                self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
            else:
                print("=======>>> NOT load_scheduler")
            self.loss_epoch = checkpoint['loss']
            self.keys = checkpoint['keys']

            # finetune
            if self.cfg.TRAIN.params.finetune:
                start_epoch = self.cfg.TRAIN.params.start_epoch
                num_epochs_remain = self.cfg.TRAIN.end_epoch - start_epoch

                lr = self.cfg.TRAIN.params.lr
                # Reset LR về 0.000001 sau khi load checkpoint
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # reset lại step của scheduler để bắt đầu từ đầu hoặc từ epoch hiện tại
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs_remain)

                # reset epoch
                self.epoch = start_epoch
            self.logger.info(f'Success: load_checkpoint: {checkpoint_filepath} \n')
            return self.epoch
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
            return -1
        except Exception as e:
            self.logger.info(f'Error: load_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
            return -1

    def transfer_scheduler_checkpoint(self, checkpoint_filepath_source, checkpoint_filepath_destination):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='transfer_scheduler_checkpoint()')
        # load to get self.scheduler from checkpoint_source
        self.load_checkpoint(checkpoint_filepath_source, is_load_scheduler=True)

        # load to get: model, optimizer, memory items
        self.load_checkpoint(checkpoint_filepath_destination, is_load_scheduler=False)
        self.save_checkpoint(checkpoint_filepath_destination)

    def train(self):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='train()')
        super().train()

    def calculate_loss_mean(self, clip, is_pseudo=False):
        """
            Tính toán các thành phần loss và gộp lại thành 1 dictionary.
            Nếu cấu hình (cfg) chỉ định loại loss nào cần sử dụng thì chỉ lấy các key tương ứng để tính tổng loss.
        """
        # clip is Tensor(bz=4, 18, 256, 256) => clip[b] is Tensor(18, 256, 256)
        num_channels = self.cfg.DATASET.channel  # 3
        num_frames = self.cfg.DATASET.num_frames  # 6
        start_past = 0
        end_past = num_channels
        start_future = (num_frames - 1) * num_channels  # (6-1)*3 = 15
        end_future = num_frames * num_channels  # 6*3 = 18

        past_frame = clip[:, start_past:end_past].to(self.device)
        input_frames = clip[:, end_past:start_future].to(self.device)
        future_frame = clip[:, start_future:end_future].to(self.device)

        predict, fea, self.keys, loss_separate, loss_compact = self.model(
            x=input_frames,
            keys=self.keys,
            train=True,
            is_pseudo=is_pseudo)

        # Tính các thành phần loss và gán vào dictionary
        predict_weights = self.cfg.TRAIN.params.predict_weights
        loss_weights = self.cfg.TRAIN.params.loss_weights

        target = {"past": past_frame, "future": future_frame}
        loss_intensity = self.loss_fn(predict, target)
        losses = {'loss_intensity_past': predict_weights[0] * loss_weights[0] * loss_intensity["past"],
                  'loss_compact_past': predict_weights[0] * loss_weights[1] * torch.mean(loss_compact["past"]),
                  'loss_separate_past': predict_weights[0] * loss_weights[2] * torch.mean(loss_separate["past"]),

                  'loss_intensity_future': predict_weights[1] * loss_weights[0] * loss_intensity["future"],
                  'loss_compact_future': predict_weights[1] * loss_weights[1] * torch.mean(loss_compact["future"]),
                  'loss_separate_future': predict_weights[1] * loss_weights[2] * torch.mean(loss_separate["future"]),
                  }

        # Lấy cấu hình loss type để xác định các loss cần sử dụng
        selected_keys = ['loss_intensity_past', 'loss_compact_past', 'loss_separate_past',
                         'loss_intensity_future', 'loss_compact_future', 'loss_separate_future']

        # Tổng hợp thành 1 total loss
        total_loss = sum(losses[k] for k in selected_keys if k in losses)
        losses['total_loss'] = total_loss

        # Properly manage tensors by deleting them when they are no longer needed
        # del output, input_frames, future_frame, clip
        return losses, fea

    def train_epoch(self, epoch):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='train_epoch()')
        self.epoch = epoch + 1

        print_screen(logger=self.logger, key='batch_size', val=self.bz)
        model_loss_type = getattr(self.cfg.MODEL.params, "loss_type", None)  # Nếu không tồn tại, trả về None
        print_screen(logger=self.logger, key='cfg.MODEL.params.loss_type', val=model_loss_type)

        loss_keys = ['total_loss',
                     'loss_intensity_past', 'loss_compact_past', 'loss_separate_past',
                     'loss_intensity_future', 'loss_compact_future', 'loss_separate_future']
        train_loss_dict = {key: AverageMeter() for key in loss_keys}

        for idx, clip in enumerate(tqdm(self.train_dataloader, desc="Processing Clips")):
            # clip is Tensor(bz=4, 15, 256, 256) => clip[b] is Tensor(15, 256, 256)
            clip = clip.to(self.device)
            losses, fea = self.calculate_loss_mean(clip)
            total_loss = losses['total_loss']
            self.optimizer.zero_grad()
            total_loss.backward()

            self.optimizer.step()

            batch_size = clip.size(0)
            # Cập nhật các loss vào AverageMeter cho từng key
            for key in loss_keys:
                loss_val = losses.get(key, 0.0)
                if isinstance(loss_val, torch.Tensor):
                    train_loss_dict[key].update(loss_val.item(), batch_size)
                else:
                    train_loss_dict[key].update(loss_val, batch_size)

            # Giải phóng bộ nhớ tạm
            del clip, total_loss, losses, fea
            torch.cuda.empty_cache()

        for key in loss_keys:
            print_screen(logger=self.logger, key=key, val=(train_loss_dict[key].avg,), format_string='{:.9f}')

        # average the loss_epoch
        self.loss_epoch = train_loss_dict['total_loss'].avg
        return self.loss_epoch

    def evaluate(self):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='evaluate()')
        super().evaluate()

    def process_video(self, idx, video):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method,
                     val=f"=========================================\n[{idx + 1}/{len(self.test_dataloader)}]")

        frame_labels = self.video_labels[idx]  # Abnormal labels (1: abnormal, 0: normal)

        # clip is Tensor(bz=4, 18, 256, 256) => clip[b] is Tensor(18, 256, 256)
        num_channels = self.cfg.DATASET.channel  # 3
        num_frames = self.cfg.DATASET.num_frames  # 6
        start_past = 0
        end_past = num_channels
        start_future = (num_frames - 1) * num_channels  # 5*3 = 15
        end_future = num_frames * num_channels  # 6*3 = 18

        # frame_psnr, frame_fea_distance = [], []
        # Khởi tạo dictionary để lưu giá trị cho mỗi frame
        frame_data = {
            i: {"past": {"psnr": 0.0, "fea_distance": 0.0},
                "future": {"psnr": 0.0, "fea_distance": 0.0}}
            for i in range(len(video))
        }

        # ==============================================
        # Iterate over clips in the video
        # ==============================================
        for f in tqdm(range(len(video) - num_frames + 1)):
            clip = torch.cat(video[f: f + num_frames], dim=1).to(self.device)
            past_frame_idx = f  # Index of past_frame
            future_frame_idx = f + (num_frames - 1)  # Index of future_frame

            past_frame = clip[:, start_past:end_past]
            input_frames = clip[:, end_past:start_future]
            future_frame = clip[:, start_future:end_future]
            target = {"past": past_frame, "future": future_frame}

            # Compute outputs and features
            predict, _, self.keys, loss_compact = self.model(
                x=input_frames, keys=self.keys, train=False)

            # Compute PSNR
            loss_intensity_past = psnr_1(predict["past"], target["past"])
            loss_intensity_future = psnr_1(predict["future"], target["future"])

            # save past_frame into frame_data
            frame_data[past_frame_idx]["past"]["psnr"] = loss_intensity_past
            frame_data[past_frame_idx]["past"]["fea_distance"] = loss_compact["past"].item()

            # save future_frame into frame_data
            frame_data[future_frame_idx]["future"]["psnr"] = loss_intensity_future
            frame_data[future_frame_idx]["future"]["fea_distance"] = loss_compact["future"].item()

            # Free memory
            del clip, past_frame, input_frames, future_frame

        # ==============================================
        # Combine values for each frame
        # ==============================================
        frame_psnr_combined = []
        frame_fea_distance_combined = []

        for i in range(len(video)):
            # past_frame
            psnr_past = frame_data[i]["past"]["psnr"]
            fea_past = frame_data[i]["past"]["fea_distance"]

            # future_frame
            psnr_future = frame_data[i]["future"]["psnr"]
            fea_future = frame_data[i]["future"]["fea_distance"]

            # final psnr
            if psnr_past is not None and psnr_future is not None:
                psnr = max(psnr_past, psnr_future)
            elif psnr_past is not None:
                psnr = psnr_past
            elif psnr_future is not None:
                psnr = psnr_future
            else:
                psnr = 0

            # final fea_distance
            if fea_past is not None and fea_future is not None:
                fea_distance = max(fea_past, fea_future)
            elif fea_past is not None:
                fea_distance = fea_past
            elif fea_future is not None:
                fea_distance = fea_future
            else:
                fea_distance = 0

            frame_psnr_combined.append(psnr)
            frame_fea_distance_combined.append(fea_distance)
        return frame_labels, frame_psnr_combined, frame_fea_distance_combined

    def test(self):
        test_type = self.cfg.TEST.test_type
        if test_type == "all":
            self.test_all()
        elif test_type == "video":
            self.test_video()
        return 1

    def test_video(self):
        # Print method header
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='test_video()')
        result = super().test()
        if result == -1:
            return -1

        # Initialize evaluation mode
        self.model.eval()
        alpha_best = self.cfg.TEST.params.alpha_best

        # Iterate over videos
        with torch.no_grad():
            for idx, video in enumerate(self.test_dataloader):
                if idx + 1 in self.cfg.TEST.plot_video_ids or -1 in self.cfg.TEST.plot_video_ids:
                    frame_labels, frame_psnr_combined, frame_fea_distance_combined = self.process_video(idx, video)

                    scores, starts, ends = self.plot_anomaly_scores_id(idx + 1, video, frame_labels,
                                                                       frame_psnr_combined, frame_fea_distance_combined,
                                                                       alpha_best)
                    if self.cfg.TEST.export_video:
                        self.plot_video_id(idx + 1, video, scores, starts, ends, alpha_best)
                    del frame_psnr_combined, frame_fea_distance_combined
        return 1

    def test_all(self):
        # Print method header
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='test_all()')
        result = super().test()
        if result == -1:
            return -1

        # Initialize evaluation mode
        self.model.eval()
        video_psnr, video_fea_distance = [], []

        # Iterate over videos
        with torch.no_grad():
            for idx, video in enumerate(self.test_dataloader):
                frame_labels, frame_psnr_combined, frame_fea_distance_combined = self.process_video(idx, video)

                # Store results for the video
                video_psnr.append(frame_psnr_combined)
                video_fea_distance.append(frame_fea_distance_combined)
                del frame_psnr_combined, frame_fea_distance_combined

        # Check video length consistency
        assert len(video_psnr) == len(self.video_labels), (
            f"Ground truth has {len(self.video_labels)} videos, but got {len(video_psnr)} detected videos!")

        # Find the best alpha
        self.get_best_alpha(video_psnr, video_fea_distance)
        return 1

    def get_best_alpha(self, video_psnr, video_fea_distance):
        best_result = MetricsResult()
        current_result = MetricsResult()
        default_result = MetricsResult()

        alpha_values, auc_values = [], []
        alpha_default = self.cfg.TEST.params.alpha_default

        for alpha in self.cfg.TEST.params.alpha_list:
            auc, fpr, tpr, optimal_idx, optimal_threshold, labels, scores = cal_auc_psnr_feas(
                video_psnr, video_fea_distance, self.video_labels, alpha)

            eer = calculate_eer(fpr=fpr, fnr=1 - tpr)
            accuracy = calculate_accuracy(fpr=fpr, tpr=tpr, idx=optimal_idx)
            f1_score = calculate_f1_score(fpr=fpr, tpr=tpr, idx=optimal_idx)

            self.logger.info('-------------------')
            print_screen(logger=self.logger, key='alpha', val=(alpha,), format_string='{:.2f}')
            print_screen(logger=self.logger, key='AUC', val=(auc * 100,), format_string='{:.2f}')

            alpha_values.append(alpha)
            auc_values.append(auc * 100)

            current_result.update(
                alpha=alpha,
                auc=auc,
                eer=eer,
                accuracy=accuracy,
                f1_score=f1_score,
                fpr=fpr,
                tpr=tpr,
                idx=optimal_idx,
                threshold=optimal_threshold,
                labels=labels,
                scores=scores
            )

            # Update best_results if AUC better
            if auc > best_result.auc:
                best_result = MetricsResult()
                best_result.update(**vars(current_result))  # copy data from current_result

            if alpha == alpha_default:
                default_result = MetricsResult()
                default_result.update(**vars(current_result))  # copy data from current_result

        # Save all values of (alpha, auc)
        alpha_values.append(best_result.alpha)
        auc_values.append(best_result.auc * 100)
        save_alpha_auc_values(alpha_values, auc_values,
                              os.path.join(self.visualization_dir_path, 'alpha_auc_values.csv'))

        save_metrics(best_result, os.path.join(self.visualization_dir_path, 'best_result.csv'))
        save_metrics(default_result, os.path.join(self.visualization_dir_path, 'default_result.csv'))

        self.visualize_test_result(best_result, default_result)
        self.log_test_result(best_result, default_result)

        return best_result.alpha

    def visualize_test_result(self, best_result, default_result):
        # draw best_AUC image (best_result)
        export_dir = make_dir_path(self.visualization_dir_path, str(best_result.alpha))
        filepath = os.path.join(export_dir, 'best_AUC.png')
        plot_auc_manual(best_result.fpr, best_result.tpr, color='darkorange', filepath=filepath)

        # draw default_AUC image (default_result)
        export_dir = make_dir_path(self.visualization_dir_path, str(default_result.alpha))
        filepath = os.path.join(export_dir, 'default_AUC.png')
        plot_auc_manual(default_result.fpr, default_result.tpr, color='darkorange', filepath=filepath)

    def log_test_result(self, best_result, default_result):
        print_screen(logger=self.logger, key='best_alpha', val=(best_result.alpha,), format_string='{:.2f}')
        print_screen(logger=self.logger, key='best_AUC', val=(best_result.auc * 100,),
                     format_string='{:.2f}')

        print_screen(logger=self.logger, key='default_alpha', val=(default_result.alpha,), format_string='{:.2f}')
        print_screen(logger=self.logger, key='default_AUC', val=(default_result.auc * 100,),
                     format_string='{:.2f}')

    def plot_anomaly_scores_id(self, video_id, video, frame_labels, frame_psnr, frame_fea_distance, best_alpha):
        export_dir = make_dir_path(os.path.join(self.visualization_dir_path, str(best_alpha)), f'{video_id:02d}')

        # Compute anomaly scores
        starts, ends = get_anomaly_rectanges(frame_labels)
        scores = calculate_anomaly_scores_MNAD(frame_psnr, frame_fea_distance, best_alpha)

        # Save and plot anomaly scores
        frame_ids_ano_scores = list(range(len(video)))
        self._save_and_plot_anomaly_scores(video_id, frame_ids_ano_scores, scores, starts, ends, export_dir)
        return scores, starts, ends

    def plot_video_id(self, video_id, video, scores, starts, ends, best_alpha):
        if video_id <= 0:
            return

        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='plot_video_id()')
        self.model.eval()

        export_dir = make_dir_path(os.path.join(self.visualization_dir_path, str(best_alpha)), f'{video_id:02d}')
        keys = self.keys

        # clip is Tensor(bz=4, 18, 256, 256) => clip[b] is Tensor(18, 256, 256)
        num_channels = self.cfg.DATASET.channel  # 3
        num_frames = self.cfg.DATASET.num_frames  # 6
        start_past = 0
        end_past = num_channels
        start_future = (num_frames - 1) * num_channels  # 5*3 = 15
        end_future = num_frames * num_channels  # 6*3 = 18

        # Logging
        self.logger.info("=========================================")
        self.logger.info(f'[{video_id}/{len(self.test_dataloader)}]')

        # Initialize variables for plotting
        frame_ids_visualize = []
        targets, predicts = [], []

        # the first (num_frames-1) in video
        for f in range(num_frames - 1):
            frame_ids_visualize.append(f)
            clip = torch.cat(video[f:f + num_frames], dim=1).to(self.device)
            past_frame = clip[:, start_past:end_past]
            future_frame = clip[:, start_past:end_past]

            target = {"past": past_frame, "future": future_frame}
            predict = {"past": past_frame, "future": future_frame}
            targets.append(target)
            predicts.append(predict)

        # Process each clip in the video
        with torch.no_grad():
            for f in tqdm(range(len(video) - num_frames + 1)):  # Sliding window over the video
                clip = torch.cat(video[f:f + num_frames], dim=1).to(self.device)
                frame_ids_visualize.append(f + num_frames - 1)

                past_frame = clip[:, start_past:end_past]
                input_frames = clip[:, end_past:start_future]
                future_frame = clip[:, start_future:end_future]
                target = {"past": past_frame, "future": future_frame}

                predict, _, keys, _ = self.model(x=input_frames, keys=keys, train=False)

                # Store future_frame and output for later plotting
                targets.append(target)
                predicts.append(predict)

                # Cleanup
                del clip, past_frame, input_frames, future_frame

        # Export visualizations (GIF/Video)
        self._export_visualizations(video_id, targets, predicts,
                                    frame_ids_visualize, scores, starts, ends, export_dir)

    def _save_and_plot_anomaly_scores(self, video_id, frame_ids, scores, starts, ends, export_dir):
        """Helper function to save and plot anomaly scores."""
        # Save anomaly scores as a graph
        anomaly_scores_graph = os.path.join(export_dir, f'anomaly_scores_video_{video_id:02d}.png')
        plot_anomaly_scores_MNAD(video_id, frame_ids, scores, starts, ends, anomaly_scores_graph)

    def _export_visualizations(self, video_id, targets, predicts,
                               frame_ids, scores, starts, ends, export_dir):
        """Helper function to export visualizations (GIF or video)."""
        if "video" in self.cfg.TEST.plot_types:
            dataset = self.cfg.DATASET.name
            method = self.cfg.MODEL.method
            plot_anomaly_heatmaps_lines_pastfuture(video_id, targets, predicts,
                                                   frame_ids, scores, starts, ends,
                                                   export_dir, dataset=dataset, method=method)

    def finetune(self):
        # Print method header
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='finetune()')
        super().finetune()