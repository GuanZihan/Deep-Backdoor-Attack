from __future__ import print_function
import pandas as pd

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def load_pretrained_model(model, pretrained_dict, wfc=True):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if wfc:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and ('fc' not in k))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_history(cls_orig_acc, clease_trig_acc, cls_trig_loss, at_trig_loss, at_epoch_list, logs_dir):
    dataframe = pd.DataFrame({'epoch': at_epoch_list, 'cls_orig_acc': cls_orig_acc, 'clease_trig_acc': clease_trig_acc,
                              'cls_trig_loss': cls_trig_loss, 'at_trig_loss': at_trig_loss})
    dataframe.to_csv(logs_dir, index=False, sep=',')

