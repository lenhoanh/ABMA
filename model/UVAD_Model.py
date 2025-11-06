from model.AbstractModel import AbstractModel
import os
import torch
from model.util.hardware import check_gpu_status
from dataset.process import get_train_dataloader, get_test_dataset
from dataset.label import LabelVideoDataset
from system.output import make_dir_path
from model.util.seed import set_seed
import random
from model.util.plot import print_screen


class UVAD_Model(AbstractModel):
    def __init__(self, cfg, device, logger, run, output_dir):
        super().__init__(cfg, device, logger, run, output_dir)
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='__init__()')

        self.train_dataloader = None
        self.test_dataloader = None
        self.video_labels = None
        self.checkpoint_filepath = None
        self.checkpoint_dir_path = None
        self.visualization_dir_path = None

        self.init_seed()
        self.init_dataset()

    def init_seed(self):
        if "seed" in self.cfg.SYSTEM:
            print_screen(logger=self.logger, key='cfg.SYSTEM.seed.use', val=self.cfg.SYSTEM.seed.use)
            if self.cfg.SYSTEM.seed.use:
                print_screen(logger=self.logger, key='cfg.SYSTEM.seed.fixed', val=self.cfg.SYSTEM.seed.fixed)
                if self.cfg.SYSTEM.seed.fixed:
                    seed = self.cfg.SYSTEM.seed.value
                else:
                    seed = random.randint(1, 2025)
                    self.cfg.SYSTEM.seed.value = seed
                set_seed(seed)
                print_screen(logger=self.logger, key='seed', val=(seed,))

                # save to neptune
                if self.run is not None:
                    self.run["init/seed"].log(seed)
        else:
            print_screen(logger=self.logger, key='cfg.SYSTEM.seed.use', val='NOT EXIST')

    def init_dataset(self):
        # get train_dataloader, self.test_dataloader, video_labels
        if self.cfg.SYSTEM.phase == 'train':
            # train_dataloader
            self.train_dataloader = get_train_dataloader(self.cfg)
            print_screen(logger=self.logger, key='len(train_dataloader) = len(train_dataset) / batch_size',
                         val=(len(self.train_dataloader),))

            # checkpoint_filepath
            self.checkpoint_filepath = os.path.join('output', 'train',
                                                    self.cfg.MODEL.name, self.cfg.MODEL.method, self.cfg.DATASET.name,
                                                    self.cfg.TRAIN.dir_name, 'checkpoint', self.cfg.TRAIN.file_name)
        elif self.cfg.SYSTEM.phase == 'test':
            # test_dataloader
            self.test_dataloader = get_test_dataset(self.cfg)
            print_screen(logger=self.logger, key='len(self.test_dataloader) = len(test_dataset) / batch_size',
                         val=(len(self.test_dataloader),))

            # labels
            self.video_labels = LabelVideoDataset(self.cfg.DATASET.name, self.cfg.DATASET.test.test_set)()
            if len(self.video_labels) == 0:
                print_screen(logger=self.logger, key='WARNING',
                             val='len(self.video_labels) == 0. Check PATH to label files.')
                return

            # checkpoint_filepath
            self.checkpoint_filepath = os.path.join('output', 'train',
                                                    self.cfg.MODEL.name, self.cfg.MODEL.method, self.cfg.DATASET.name,
                                                    self.cfg.TEST.dir_name, 'checkpoint', self.cfg.TEST.file_name)
        elif self.cfg.SYSTEM.phase == 'evaluate':
            # train_dataloader
            self.train_dataloader = get_train_dataloader(self.cfg)
            print_screen(logger=self.logger, key='len(train_dataloader) = len(train_dataset) / batch_size',
                         val=(len(self.train_dataloader),))

            # checkpoint_filepath
            self.checkpoint_filepath = os.path.join('output', 'train',
                                                    self.cfg.MODEL.name, self.cfg.MODEL.method, self.cfg.DATASET.name,
                                                    self.cfg.EVALUATE.dir_name, 'checkpoint',
                                                    self.cfg.EVALUATE.file_name)
        elif self.cfg.SYSTEM.phase == 'finetune':
            # train_dataloader
            self.train_dataloader = get_train_dataloader(self.cfg)
            print_screen(logger=self.logger, key='len(train_dataloader) = len(train_dataset) / batch_size',
                         val=(len(self.train_dataloader),))

            # checkpoint_filepath
            self.checkpoint_filepath = os.path.join('output', 'train',
                                                    self.cfg.MODEL.name, self.cfg.MODEL.method, self.cfg.DATASET.name,
                                                    self.cfg.FINETUNE.dir_name, 'checkpoint',
                                                    self.cfg.FINETUNE.file_name)
        print_screen(logger=self.logger, key='checkpoint_filepath', val=self.checkpoint_filepath)

        # checkpoint dir
        self.checkpoint_dir_path = make_dir_path(self.output_dir, self.cfg.SYSTEM.output.checkpoint_dir)
        print_screen(logger=self.logger, key='checkpoint_dir_path', val=self.checkpoint_dir_path)

        # visualization dir
        self.visualization_dir_path = make_dir_path(self.output_dir, self.cfg.SYSTEM.output.visualization_dir)
        print_screen(logger=self.logger, key='visualization_dir_path', val=self.visualization_dir_path)

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

    def load_checkpoint(self, checkpoint_filepath):
        print_screen(logger=self.logger, key=self.cfg.MODEL.method, val='load_checkpoint()')
        return super().load_checkpoint(checkpoint_filepath)

    def train_epoch(self, epoch):
        print_screen(logger=self.logger, key='UVAD_Model', val='train_epoch()')
        # =========================
        # Activate train mode
        # =========================
        self.epoch = epoch + 1
        self.model = self.model.to(self.device)
        self.model.train()
        loss_epoch = 0
        return loss_epoch

    def train(self):
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        self.logger.info('>>>>>>>>>> TRAINING PROCESS >>>>>>>>>> ')
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        print_screen(logger=self.logger, key='UVAD_Model', val='train()')

        begin_epoch = self.cfg.TRAIN.begin_epoch
        end_epoch = self.cfg.TRAIN.end_epoch
        if self.cfg.TRAIN.resume:
            # ==================================
            self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
            self.logger.info(">>>>>>>> RESUME TRAINING: >>>>>>>>>>>")
            self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
            # ==================================
            epoch = self.load_checkpoint(self.checkpoint_filepath)
            if epoch == -1:
                return
            begin_epoch = epoch
        # ==================================
        # Training process
        # ==================================
        self.model = self.model.to(self.device)

        self.model.train()
        on_timer = True
        starter_train = 0
        ender_train = 0
        if on_timer:
            starter_train, ender_train = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_train.record()

        for epoch in range(begin_epoch, end_epoch):
            self.logger.info(f'epoch[{epoch + 1}/{end_epoch}]')
            print(f'epoch[{epoch + 1}/{end_epoch}]')
            starter_epoch = 0
            ender_epoch = 0
            if on_timer:
                starter_epoch, ender_epoch = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter_epoch.record()

            # Training an epoch
            self.train_epoch(epoch)

            # Print training time per epoch
            if on_timer:
                ender_epoch.record()
                torch.cuda.synchronize()  # Waits for everything to finish running
                total_seconds_epoch = starter_epoch.elapsed_time(ender_epoch) * 1e-3  # millisecond to second

                # Calculate minutes and remaining seconds
                hours = total_seconds_epoch // 3600
                minutes = (total_seconds_epoch % 3600) // 60
                seconds = total_seconds_epoch % 60
                self.logger.info(f'Time for epoch: {hours} hours, {minutes} minutes and {seconds} seconds')
                print(f'Time for epoch: {hours} hours, {minutes} minutes and {seconds} seconds')
            print("========================")
            # Change learning_rate
            if self.cfg.TRAIN.lr_scheduler.use:
                self.scheduler.step()
                cur_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info('lr_scheduler: {lr:.6f}'.format(lr=cur_lr))
                print('lr_scheduler: {lr:.6f}'.format(lr=cur_lr))

            # Save the trained model by sequence
            if (epoch + 1) % self.cfg.TRAIN.save_freq == 0:
                checkpoint_filepath = os.path.join(self.checkpoint_dir_path, f'epoch_{epoch + 1}.pth')
                self.save_checkpoint(checkpoint_filepath)

            # Check status of GPU, RAM usage
            check_gpu_status(epoch + 1)
            print("================================================")

        # ==================================
        # Save the final trained model
        # ==================================
        if on_timer:
            ender_train.record()
            torch.cuda.synchronize()  # Waits for everything to finish running
            total_seconds_train = starter_train.elapsed_time(ender_train) * 1e-3  # millisecond to second

            # Calculate hours, minutes, and seconds
            hours = total_seconds_train // 3600
            minutes = (total_seconds_train % 3600) // 60
            seconds = total_seconds_train % 60
            self.logger.info(f'Time for train:{hours} hours, {minutes} minutes, and {seconds} seconds')
            print(f'Time for train:{hours} hours, {minutes} minutes, and {seconds} seconds')

        checkpoint_filepath = os.path.join(self.checkpoint_dir_path, f'epoch_{end_epoch}.pth')
        self.save_checkpoint(checkpoint_filepath)

    def test(self):
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        self.logger.info('>>>>>>>>>> TESTING PROCESS >>>>>>>>>> ')
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        print_screen(logger=self.logger, key='UVAD_Model', val='test()')
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(self.checkpoint_filepath)
        if epoch == -1:
            return -1
        self.model = self.model.to(self.device)

        # Activate evaluation mode
        self.model.eval()
        return 1

    def evaluate(self):
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        self.logger.info('>>>>>>>>>> EVALUATE PROCESS >>>>>>>>>> ')
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        print_screen(logger=self.logger, key='UVAD_Model', val='evaluate()')
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(self.checkpoint_filepath)
        if epoch == -1:
            return -1
        self.model = self.model.to(self.device)

        # Activate evaluation mode
        self.model.eval()
        return 1

    def finetune(self):
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        self.logger.info('>>>>>>>>>> FINE-TUNE PROCESS >>>>>>>>>> ')
        self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
        print_screen(logger=self.logger, key='UVAD_Model', val='finetune()')
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(self.checkpoint_filepath)
        if epoch == -1:
            return -1
        self.model = self.model.to(self.device)

        return 1
