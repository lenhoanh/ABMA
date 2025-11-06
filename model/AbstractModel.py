from abc import ABC, abstractmethod
import torch
from model.util.plot import print_screen


class AbstractModel(ABC):
    def __init__(self, cfg, device, logger, run, output_dir):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.run = run
        self.output_dir = output_dir
        self.logger.info('Call: AbstractModel.__init__()')
        self.name = cfg.MODEL.name
        self.type = cfg.MODEL.type

        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.loss_epoch = 0.0

    @abstractmethod
    def save_model(self, model_filepath):
        print_screen(logger=self.logger, key='AbstractModel', val='save_model()')
        try:
            torch.save(self.model, model_filepath)
            self.logger.info('Success: Model saved')
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            self.logger.info('Error: Model not saved!!!')
            return 0
        return 1

    @abstractmethod
    def load_model(self, model_filepath):
        print_screen(logger=self.logger, key='AbstractModel', val='load_model()')
        try:
            self.model = torch.load(model_filepath)
            return self.model
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            return None

    @abstractmethod
    def save_model_state_dict(self, model_filepath):
        print_screen(logger=self.logger, key='AbstractModel', val='save_model_state_dict()')
        try:
            torch.save(self.model.state_dict(), model_filepath)
            self.logger.info('Success: Model_state_dict saved')
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            self.logger.info('Error: Model_state_dict not saved!!!')
            return 0
        return 1

    @abstractmethod
    def load_model_state_dict(self, model_filepath):
        print_screen(logger=self.logger, key='AbstractModel', val='load_model_state_dict()')
        try:
            state_dict = torch.load(model_filepath)
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
                self.model.load_state_dict(state_dict)
            else:
                self.model.module.load_state_dict(state_dict)
            return self.model
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            return None

    @abstractmethod
    def save_checkpoint(self, checkpoint_filepath):
        print_screen(logger=self.logger, key='AbstractModel', val='save_checkpoint()')
        print_screen(logger=self.logger, key='checkpoint_filepath', val=checkpoint_filepath)
        try:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'loss': self.loss_epoch
            }, checkpoint_filepath)
            self.logger.info(f'Success: save_checkpoint: {checkpoint_filepath}\n')
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            self.logger.info(f'Error: save_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
        return 1

    @abstractmethod
    def load_checkpoint(self, checkpoint_filepath):
        print_screen(logger=self.logger, key='AbstractModel', val='load_checkpoint()')
        print_screen(logger=self.logger, key='checkpoint_filepath', val=checkpoint_filepath)
        try:
            # The map_location parameter specifies where the tensors will be loaded, 
            # particularly useful when loading models saved on a different device (e.g., CUDA GPU)
            # onto a different device (e.g., CPU or a different GPU).
            checkpoint = torch.load(checkpoint_filepath,
                                    map_location=self.cfg.SYSTEM.device)
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
            self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
            self.loss_epoch = checkpoint['loss']
            self.logger.info(f'Success: load_checkpoint: {checkpoint_filepath} \n')
            return self.epoch
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
            self.epoch = -1
            return -1
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            self.logger.info(f'Error: load_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
            self.epoch = -1
            return -1
