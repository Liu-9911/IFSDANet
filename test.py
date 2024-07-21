import sys
sys.path.append('../..')
from UDL.AutoDL import TaskDispatcher
from UDL.AutoDL.trainer import main

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='GF-CSTNet')
    cfg.eval = True
    cfg.resume_from = "../pretrained-model/WV3/gfcsp.pth"
    cfg.workflow = [('val', 1)]
    cfg.dataset = {'val': 'Test_wv3'}
    print(TaskDispatcher._task.keys())
    main(cfg)
