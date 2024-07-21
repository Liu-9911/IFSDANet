import sys
sys.path.append('../..')
from UDL.AutoDL import TaskDispatcher
from UDL.AutoDL.trainer import main

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='GF-CSPNet')
    cfg.workflow = [('train',50), ('val', 1)]  # 默认是train = 50,训练50个epoch验证一次
    print(TaskDispatcher._task.keys())
    main(cfg)

