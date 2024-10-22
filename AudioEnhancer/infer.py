#encoding:utf-8
import os
import importlib
import datetime
from utils.config import parser

args = parser.parse_args() #解析外部参数

def main(args):
    args.config = os.path.join('task', args.task, 'config', args.config)
    print('cur config:', args.config)
    import_module_path = 'task.' + args.task + '.' +  args.runner
    print('cur runner:', import_module_path)

    engine = importlib.import_module(import_module_path)
    runner = engine.Runner(args)
    runner.infer()






if __name__ == "__main__":
    print('hello SpeechEnhancer')
    main(args)



