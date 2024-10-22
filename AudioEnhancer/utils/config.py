#encoding:utf-8
import argparse


parser = argparse.ArgumentParser() #也可以直接()，不用description
parser.add_argument('--task', type=str, help='audio task')
parser.add_argument('--config', type=str, help='config yaml')
parser.add_argument('--runner', type=str, help='exec file')
parser.add_argument('--init_ckpt', type=str, help='finetune pt')
parser.add_argument('--savedir', type=str, help='savedir pt')


