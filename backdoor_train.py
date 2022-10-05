from models.selector import *
from utils.util import *
from data_loader import *
from torch.utils.data import DataLoader
from config import get_arguments
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR

perm1 = set()
perm2 = set()

def compute_loss_value(opt, poisoned_data, model_ascent):
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record_original = []
    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )
    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()
        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
        losses_record_original.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record_original))
    return losses_idx

def isolate_data(poisoned_data, losses_idx, ratio):
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0


    example_data_loader = DataLoader(dataset=poisoned_data,
                      batch_size=1,
                      shuffle=False,
                      )
    global perm2
    
    perm = losses_idx[0: int(len(losses_idx) * ratio)]
    perm2 = set(perm)

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        img = img.squeeze()
        target = target.squeeze()
        img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        # Filter the examples corresponding to losses_idx
        if idx in perm:
            isolation_examples.append((img, target))
            cnt += 1
        else:
            other_examples.append((img, target))

    print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    print('Finish collecting {} other examples: '.format(len(other_examples)))


def train_step(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    clean_loss = AverageMeter()
    perm_loss = AverageMeter()

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)
 
        loss_ascent = criterion(output, target)

        
        optimizer.zero_grad()
        loss_ascent.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_ascent.item(), img.size(0))

        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'Loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'Loss_clean:{losses_clean.val:.4f}({losses_clean.avg:.4f})  '
                  'Loss_bad:{losses_bad.val:.4f}({losses_bad.avg:.4f})  '
                  'Prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'Prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, losses_clean=clean_loss ,losses_bad=perm_loss,top1=top1, top5=top5))




def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch, mode):
    test_process = []
    
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()
    # idx batch idx
    for idx, (img, target) in enumerate(test_clean_loader, start=1):
      
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
        
        
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        # print(img.shape)
        # input()
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
          output = model_ascent(img)
        # img.requires_grad = True
          
        
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))


    acc_bd = [top1.avg, top5.avg, losses.avg]

    print('[Mode] {} [Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(mode, acc_clean[0], acc_clean[2]))
    print('[Mode] {} [Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(mode, acc_bd[0], acc_bd[2]))


    return acc_clean, acc_bd

def addsuffix(opt):
  if opt.dataset == "CIFAR10":
    return ""
  elif (opt.dataset == 'GTSRB'):
    return "_gtsrb"
  elif opt.dataset == "Cifar100":
    return "_Cifar100"
  elif opt.dataset == "FMNIST":
    return "_mnist"


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    model_ascent, _ = select_model(dataset=opt.dataset,
                           model_name=opt.model_name,
                           pretrained=False,
                           pretrained_models_path="",
                           n_classes=opt.num_class)
    if opt.cuda:
        model_ascent.to(opt.device)

    print('finished model init...')

    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


    print('----------- Data Initialization --------------')
    global perm1
    poisoned_data = None
    if opt.load_fixed_data:
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc.
        if opt.trigger_type == "dba":
          poisoned_data = np.load("./data/dba/poisoned_data_dba.npy", allow_pickle=True)
          perm_index_fixed = np.load("./data/dba/perm_index_dba.npy", allow_pickle=True)

        
        tf_train = transforms.Compose([transforms.ToTensor()])
        
  
        poisoned_data = Dataset2(poisoned_data, tf_train)
        
        poisoned_data_loader = DataLoader(dataset=poisoned_data,
                          batch_size=opt.batch_size,
                            shuffle=False)
        dataset=Dataset2(poisoned_data, tf_train)
        
        perm1 = set(perm_index_fixed)
    else:
        poisoned_data, poisoned_data_loader, perm, bad_data, clean_data = get_backdoor_loader(opt)
        np.save("perm_index.npy", np.array(perm))
        perm1 = set(perm)
    
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if opt.load_fixed_data:
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc.
        if opt.trigger_type == "dba":
          poisoned_data_test = np.load("./data/dba/poisoned_data_dba_only" + addsuffix(opt) + ".npy", allow_pickle=True)
          other_test = np.load("./data/dba/poisoned_data_dba_other" + addsuffix(opt) + ".npy", allow_pickle=True)
          testset_poisoned = np.load("./data/dba/poisoned_data_dba_only_test" + addsuffix(opt) + ".npy", allow_pickle=True)
        
        
        poisoned_data_test = Dataset2(poisoned_data_test, tf_test)
        test_bad_loader_trainset = DataLoader(dataset=poisoned_data_test,
                          batch_size=opt.batch_size,
                            shuffle=False)
                          
        test_clean_loader_trainset = DataLoader(dataset=Dataset2(other_test, tf_test),
                          batch_size=opt.batch_size,
                            shuffle=False)
        test_clean_loader_testset, _ = get_test_loader(opt)
        test_bad_loader_testset = DataLoader(dataset=Dataset2(testset_poisoned, tf_test),
                          batch_size=opt.batch_size,
                            shuffle=False)
        
    else:
        test_clean_loader_testset, test_bad_loader_testset = get_test_loader(opt)
  
        test_bad_loader_trainset = bad_data 
        test_clean_loader_trainset = clean_data

    best_clean_testset = 0
    Ip_dict = {}

    print('----------- Train Initialization --------------')
            
    for epoch in range(0, opt.tuning_epochs):

        # train every epoch
        if epoch == 0:
            test(opt, test_clean_loader_testset, test_bad_loader_testset, model_ascent,
                                         criterion, epoch + 1, mode="test")


        train_step(opt, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        scheduler.step()
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))
        
        # evaluate on training set
        print('testing the ascended model on training dataset......')
        print("clean ", len(test_clean_loader_trainset))
        print("bad ", len(test_bad_loader_trainset))
        acc_clean_trainset, acc_bad_trainset = test(opt, test_clean_loader_trainset, test_bad_loader_trainset, model_ascent, criterion, epoch + 1, mode="train")

        # evaluate on testing set
        print('testing the ascended model on testing dataset......')
        print("clean ", len(test_clean_loader_testset))
        print("bad ", len(test_bad_loader_testset))
        acc_clean_testset, acc_bad_testset = test(opt, test_clean_loader_testset, test_bad_loader_testset, model_ascent, criterion, epoch + 1, mode="test")

        # compute the loss value and isolate data at iteration 20
        if epoch < 5:
          print('----------- Calculate loss value per example -----------')
          losses_idx = compute_loss_value(opt, poisoned_data, model_ascent)
          print('----------- Collect isolation data -----------')
          IPs = []
          for i in [opt.inject_portion]:
            isolate_data(poisoned_data, losses_idx, i)
            global perm2
            intersection = perm1 & perm2
            ip = len(intersection) / len(perm2)
            IPs.append(ip)
            print("[Attack] {} [isolation_ratio] {:.2f} [Isolation Precision] {:.2f}".format(opt.trigger_type, i, ip))
          Ip_dict[epoch] = IPs
          
        is_best = True
        if opt.save:
            # remember best precision and save checkpoint
            if acc_clean_testset[0] >= best_clean_testset:
              best_bad_acc = acc_bad_testset[0]
              best_clean_testset = acc_clean_testset[0]
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_ascent.state_dict(),
                'bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, epoch, is_best, opt)

    return poisoned_data, model_ascent, best_clean_testset, best_bad_acc, Ip_dict


def save_checkpoint(state, epoch, opt):
    filepath = str(opt.model_name) + r'-tuning_epochs{}.tar'.format(str(epoch))
    output = open(filepath, mode="wb")
    torch.save(state, output)
    print('[info] Finish saving the model' + filepath)

 

def main():
  opt = get_arguments().parse_args() 
  opt.save = True
  print('attack: ' + str(opt.trigger_type) + " injection ratio: " + str(opt.inject_portion))

  poisoned_data, ascent_model, ca_best, asr_best, Ip_dict = train(opt)


if (__name__ == '__main__'):
    main()