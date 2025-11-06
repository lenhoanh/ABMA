import os
import time
import sys
import torch
import yaml
import argparse
import configparser
from yacs.config import CfgNode
import torch.backends.cudnn as cudnn

from system.output import make_logger, make_dir_path

sys.path.append('../')

# List of ABMA_PA (Pseuo-Anomalies) method
ABMA_PA_methods = [
    'ABMA_SF', 'ABMA_KF_NP', 'ABMA_RF', 'ABMA_Noise', 'ABMA_Patch',
    'ABMA_Occlusion', 'ABMA_Illumination',
    'ABMA_SF_Patch', 'ABMA_SF_Illumination', 'ABMA_SF_Occlusion',
    'BMA_SF', 'BMA_KF_NP',
    'AMA_SF'
]


def get_model(cfg, device, logger, run, output_dir):
    model = None

    if cfg.MODEL.method == 'ABMA':
        from model.ABMA.ABMA import ABMA
        model = ABMA(cfg, device, logger, run, output_dir)
    elif cfg.MODEL.method in ABMA_PA_methods:
        from model.ABMA.ABMA_PA import ABMA_PA
        model = ABMA_PA(cfg, device, logger, run, output_dir)

    return model


def train(cfg, device, logger, run, output_dir):
    model = get_model(cfg, device, logger, run, output_dir)
    if model is not None:
        model.train()
    if run is not None:
        run.stop()


def finetune(cfg, device, logger, run, output_dir):
    model = get_model(cfg, device, logger, run, output_dir)
    if model is not None:
        model.finetune()
    if run is not None:
        run.stop()


def test(cfg, device, logger, run, output_dir):
    model = get_model(cfg, device, logger, run, output_dir)
    if model is not None:
        model.test()
    if run is not None:
        run.stop()


def evaluate(cfg, device, logger, run, output_dir):
    model = get_model(cfg, device, logger, run, output_dir)
    if model is not None:
        model.evaluate()
    if run is not None:
        run.stop()


# ===========================================================
# args, device
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='VAD')

    parser.add_argument("--model", help='ABMA',
                        default='ABMA', type=str)
    parser.add_argument("--method", help='ABMA variants: ABMA, ABMA_SF, ...',
                        default='ABMA_SF', type=str)
    parser.add_argument("--phase", help='train or test',
                        default='train', type=str)
    parser.add_argument("--dataset", help='ped2, avenue, or shanghaitech',
                        default='ped2', type=str)
    parser.add_argument("--device", help='cuda:0, cuda:1, cpu',
                        default='cuda:0', type=str)

    '''
    https://github.com/rbgirshick/yacs
    Now override from a list (opts could come from the command line)
    '''
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # Read args from command line
    args = parser.parse_args()

    # Merge "cfg" from "cfg_filepath" and "args.opts"
    cfg_filepath = os.path.join(os.getcwd(), 'system', args.model, args.method, f'{args.dataset}.yaml')
    print(f'======>>> cfg_filepath = {cfg_filepath}')

    # Create a new CfgNode with modifications: SYSTEM.phase, DATASET.name
    modified_cfg = CfgNode()
    modified_cfg.SYSTEM = CfgNode()
    modified_cfg.SYSTEM.phase = args.phase
    modified_cfg.SYSTEM.device = args.device

    modified_cfg.DATASET = CfgNode()
    modified_cfg.DATASET.name = args.dataset

    modified_cfg.MODEL = CfgNode()
    modified_cfg.MODEL.name = args.model
    modified_cfg.MODEL.method = args.method

    # Load the configuration file
    with open(cfg_filepath, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Create a new CfgNode: cfg is the original configuration
    cfg = CfgNode(config)

    # Merge the modified CfgNode with the original configuration
    cfg.merge_from_other_cfg(modified_cfg)

    # Merge other config options with the original configuration
    # args.opts: SYSTEM.device, SYSTEM.neptune.use, TRAIN.resume
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    return cfg, cfg_filepath, args.opts


def setup_device(cfg, logger):
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available >= 1:
        logger.info(f"The number of GPUs available = {num_gpus_available}")

        # cudnn sẽ được sử dụng nếu có sẵn (trên GPU), giúp tối ưu hóa và tăng tốc quá trình tính toán.
        cudnn.enabled = cfg.SYSTEM.cudnn.enabled  # True
        # PyTorch bật chế độ tối ưu hóa của cudnn, tìm kiếm thuật toán tốt nhất cho từng kích thước của đầu vào.
        cudnn.benchmark = cfg.SYSTEM.cudnn.benchmark  # True
        # PyTorch không bắt buộc phải sử dụng các thuật toán nhất quán trong cudnn
        cudnn.determinstic = cfg.SYSTEM.cudnn.deterministic  # False

        device = torch.device(cfg.SYSTEM.device)  # "cuda:0"
        logger.info(f'device = {device}')
    else:
        logger.info("GPU unavailable")
        device = torch.device("cpu")
    return device


def setup_neptune(cfg, cfg_filepath, args_opts, output_dir):
    run = None
    neptune_config_filepath = os.path.join(os.getcwd(), 'system', 'neptune.config')
    if os.path.isfile(neptune_config_filepath) and cfg.SYSTEM.neptune.use:
        neptune_config = configparser.RawConfigParser()
        neptune_config.read(neptune_config_filepath)

        # get api_token and neptune_project
        api_token = neptune_config.get('neptune', 'api_token')
        neptune_project = neptune_config.get('neptune', 'neptune_project')

        # convert [args_opts] list to [args_dict] dictionary
        args_dict = {args_opts[i]: args_opts[i + 1] for i in range(0, len(args_opts), 2)}

        # define params dictionary
        params = {"output_dir": output_dir, "phase": cfg.SYSTEM.phase,
                  "device": cfg.SYSTEM.device, "model": cfg.MODEL.name, "method": cfg.MODEL.method,
                  "dataset": cfg.DATASET.name,
                  }

        # merge args_dict with params
        params.update(args_dict)

        # Upload params to neptune.ai
        try:
            run = neptune.init_run(
                project=neptune_project,
                api_token=api_token,
                name=cfg.MODEL.method + "_" + cfg.DATASET.name
            )
            run["parameters"] = params
            run["config"].upload(cfg_filepath)
        except Exception as e:
            print(f'Error: {str(e)}')
    return run


def make_output_dir(cfg):
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')  # 2023_08_07_10_34
    if cfg.SYSTEM.phase == 'train':
        # Get the current working directory ("/home/asus/DATA/VAD")
        root_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                                cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name)
        dir_name = f'session_{time_str}'
        make_dir_path(root_dir, dir_name)
        output_dir = os.path.join(root_dir, dir_name)
        return output_dir
    if cfg.SYSTEM.phase == 'test':
        root_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                                cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                cfg.TEST.dir_name)
        dir_name = f'session_{time_str}'
        make_dir_path(root_dir, dir_name)
        output_dir = os.path.join(root_dir, dir_name)
        return output_dir
    if cfg.SYSTEM.phase == 'evaluate':
        root_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                                cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                cfg.EVALUATE.dir_name)
        dir_name = f'session_{time_str}'
        make_dir_path(root_dir, dir_name)
        output_dir = os.path.join(root_dir, dir_name)
        return output_dir


# ===========================================================
# ===========================================================
def run_train(cfg, cfg_filepath, args_opts):
    cfg.SYSTEM.phase = 'train'

    # 2. Create output directory
    if cfg.TRAIN.resume:
        output_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                                  cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                  cfg.TRAIN.dir_name)
        print("Reusing output_dir = ", output_dir)
    else:
        output_dir = make_output_dir(cfg)
        print("Creating a new output_dir = ", output_dir)

    # 3. Create logger to print results into a text file (.log)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, args_opts, output_dir)

    # 6. Training process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    train(cfg, device, logger, run, output_dir)
    return output_dir


def run_finetune(cfg, cfg_filepath, args_opts):
    cfg.SYSTEM.phase = 'finetune'

    # 2. Create output directory
    output_dir = make_output_dir(cfg)
    print("Creating a new output_dir = ", output_dir)

    # 3. Create logger to print results into a text file (.log)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, args_opts, output_dir)

    # 6. Training process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    finetune(cfg, device, logger, run, output_dir)
    return output_dir


def run_test(cfg, cfg_filepath, args_opts):
    cfg.SYSTEM.phase = 'test'
    # 2. Create output directory
    output_dir = make_output_dir(cfg)
    print("Creating a new output_dir = ", output_dir)

    # 3. Create logger to print results into a text file (.log)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, args_opts, output_dir)

    # 6. Testing process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    test(cfg, device, logger, run, output_dir)
    return output_dir


def run_evaluate(cfg, cfg_filepath, args_opts):
    cfg.SYSTEM.phase = 'evaluate'
    # 2. Create output directory
    output_dir = make_output_dir(cfg)

    # 3. Create logger to print results into a text file (.log)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, args_opts, output_dir)

    # 6. Testing process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    evaluate(cfg, device, logger, run, output_dir)


def main():
    # SYSTEM: cfg (config), output directory, logger, tensorboard, device ...
    # 1. Read file config + args from command line
    cfg, cfg_filepath, args_opts = parse_args()
    # Set the time to wait (in seconds)
    wait_time = 10  # Change this to your desired wait time
    if cfg.SYSTEM.phase == 'train':
        for i in range(cfg.SYSTEM.num_run):
            output_dir = run_train(cfg, cfg_filepath, args_opts)

            # Wait before stopping the run (neptune)
            time.sleep(wait_time)

            # After training, autorun test to print the result...
            print("Training completed!!! Now starting testing...")

            # Update args_opts for TEST (`cfg.TEST.dir_name`)
            args_opts.extend(["TEST.dir_name", output_dir])
            # args_opts.extend(["TEST.file_name", f'epoch_{cfg.TRAIN.end_epoch}.pth'])
            cfg.merge_from_list(args_opts)
            run_test(cfg, cfg_filepath, args_opts)

            # Wait before stopping the run (neptune)
            time.sleep(wait_time)

    elif cfg.SYSTEM.phase == 'test':
        run_test(cfg, cfg_filepath, args_opts)
    elif cfg.SYSTEM.phase == 'evaluate':
        run_evaluate(cfg, cfg_filepath, args_opts)


if __name__ == '__main__':
    """
    #===================================================
    # ABMA
    #===================================================
    python main.py --model ABMA --method ABMA --phase train --dataset ped2 --device cuda:0
    python main.py --model ABMA --method ABMA_SF --phase train --dataset ped2 --device cuda:0
    """
    main()
