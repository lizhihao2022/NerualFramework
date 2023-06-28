import argparse
from models import ALL_MODELS


parser = argparse.ArgumentParser(description='MODEL NAME')
parser.add_argument("--mode", type=str, choices=["train", "valid", "test"], default="train")
# Model Parameters
parser.add_argument("--model", type=str, choices=list(ALL_MODELS.keys()), default="MLP")
parser.add_argument("--layers", type=int, default=4, help="layer num of NN")
parser.add_argument("--dim", type=int, default=64, help='dimensionality of features')
# Log Parameters
parser.add_argument("--log", type=bool, default=True, help="enable logging")
parser.add_argument("--log_dir", type=str, default="./logs", help="Dir for logs")
parser.add_argument("--wandb", type=bool, default=False, help="enable wandb")
parser.add_argument("--wandb_project", type=str, default="", help="wandb project name")
# Training Parameters
parser.add_argument("--random_seed", type=int, default=2020)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--device", type=int, default=4)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--eval_freq", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=1024, help='mini-batch size')
# Optimizer Parameters
parser.add_argument("--optimizer", type=str, choices=['Adam', 'SGD'], default='Adam')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--momentum", type=float, default=0.95)
# Dataset Parameters
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--dataset", type=str, choices=["DATASET"], default="DATASET")
parser.add_argument("--subset", type=bool, default=False)
parser.add_argument("--train_ratio", type=float, default=0.6)
parser.add_argument("--valid_ratio", type=float, default=0.2)
parser.add_argument("--test_ratio", type=float, default=0.2)
