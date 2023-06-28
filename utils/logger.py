import logging
import os
from datetime import datetime


def get_log_dir_path(model, dataset, path):
    date = datetime.now().strftime("%m_%d")
    time = datetime.now().strftime("_%H_%M_%S")
    dir_path = os.path.join(path, dataset, date, model + time)
    os.makedirs(dir_path)
    dir_name = date + "_" + model + time
    return dir_path, dir_name


def set_up_logger(args):
    log_dir, dir_name = get_log_dir_path(args.model, args.dataset, args.log_dir)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(log_dir, "train.log")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(log_dir))

    return log_dir, dir_name
