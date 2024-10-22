#encoding:utf-8
import os
import importlib
import datetime

import torch

from utils.config import parser

args = parser.parse_args() #解析外部参数
cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def train(args, device):
    args.config = os.path.join('task', args.task, 'config', args.config)
    print('cur config:', args.config)
    import_module_path = 'task.' + args.task + '.' +  args.runner
    print('cur runner:', import_module_path)
    args.device = device
    args.savedir = os.path.join(os.getcwd(), 'savedir', args.task, str(cur_time))

    os.makedirs(args.savedir, exist_ok=True)
    engine = importlib.import_module(import_module_path)
    runner = engine.Runner(args)
    runner.train()


if __name__ == "__main__":
    print("是否可用：", torch.cuda.is_available())  # 查看GPU是否可用
    print("GPU数量：", torch.cuda.device_count())  # 查看GPU数量
    print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
    print("GPU索引号：", torch.cuda.current_device())  # 查看GPU索引号

    # # 选择特定的GPU，如果是多GPU则选择特定的ID
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 选择特定的GPU，如果是多GPU则选择特定的ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(args, device)



