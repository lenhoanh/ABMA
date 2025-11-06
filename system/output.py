import os
import time
import logging


def make_output_dir(cfg):
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')  # 2023_08_07_10_34
    output_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                              cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                              f'session_{time_str}')
    return output_dir


def make_logger(output_dir: str, logger_dir_name: str):
    # 1. create output log directory
    log_dir_path = make_dir_path(output_dir, logger_dir_name)

    # 2. create log file and set up the basic of the logger
    log_file = os.path.join(log_dir_path, 'logger.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)
    print(f'=> Creating the [log_file]: {log_file}')

    # logger
    logger = logging.getLogger('PIL')
    fmt = '%(asctime)-15s:%(message)s'  # format logger
    datefmt = '%Y-%m-%d-%H:%M'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    level = logging.INFO
    logger.setLevel(level)

    # console
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    # file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # addHandlder: console, file_handler
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger, log_file


def make_dir_path(root_dir: str, dir_name: str):
    dir_path = os.path.join(root_dir, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f'=> Creating a new directory: {dir_path}')
    else:
        print(f"- The directory [{dir_path}] existed!!!")

    if not os.path.exists(dir_path):
        raise Exception('Something wrong in creating dir_path: {}'.format(dir_path))
    return dir_path
