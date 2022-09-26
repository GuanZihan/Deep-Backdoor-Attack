from models.selector import *
from utils.util import *
from data_loader import *
from torch.utils.data import DataLoader
from config import get_arguments
from tqdm import tqdm
import gc
import numpy as np
import json
from torch.optim.lr_scheduler import StepLR
import logging
# from apex import amp



losses_clean_train = []
losses_bad_train = []
losses_clean_test = []
losses_bad_test = []
backdoor_losses_record = []
clean_losses_record = []
perm1 = set()
perm2 = set()

def compute_loss_value(opt, poisoned_data, model_ascent):
    # Calculate loss value per example
    # Define loss function
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
    labels = []
    features = None

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        # img = torch.swapaxes(img, 1, 2)
        feat_list = []

        for i, element in enumerate(target):
            labels.append(element)
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()
        def hook(module, input, output): 
            feat_list.append(output.clone().detach())
        with torch.no_grad():
            # handle=model_ascent.avgpool.register_forward_hook(hook)
            output = model_ascent(img)
            loss = criterion(output, target)
            # feat = torch.flatten(feat_list[0], 1)  
            # handle.remove()
        
        # current_features = feat.cpu().numpy()
        # if features is not None:
        #     features = np.concatenate((features, current_features))
        # else:
        #     features = current_features
        
        losses_record_original.append(loss.item())
    
    
    for idx, i in enumerate(losses_record_original):
      if idx in perm1:
        backdoor_losses_record.append(i)
      else:
        clean_losses_record.append(i)


    losses_idx = np.argsort(np.array(losses_record_original))

    # Show the top 10 loss values
    losses_record_arr = np.array(losses_record_original)

    print('Top ten loss value:', losses_record_arr[losses_idx[:10]])
    # np.save("losses_record_arr", losses_record_arr)
    # print('Backdoor Loss', losses_record_arr[perm1])
    return losses_idx, features, labels

def isolate_data(opt, poisoned_data, losses_idx, ratio):
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    #ratio = opt.isolation_ratio

    tf_train = transforms.Compose([transforms.ToTensor()
                                  ])

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

    # Save data
    if opt.save:
        data_path_isolation = os.path.join(opt.isolate_data_root, "{}_isolation{}%_examples.npy".format(opt.model_name,opt.isolation_ratio * 100))
        data_path_other = os.path.join(opt.isolate_data_root, "{}_other{}%_examples.npy".format(opt.model_name,100 - opt.isolation_ratio * 100))
        np.save(data_path_isolation, isolation_examples)
        np.save(data_path_other, other_examples)

    print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    print('Finish collecting {} other examples: '.format(len(other_examples)))


def train_step(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    clean_loss = AverageMeter()
    perm_loss = AverageMeter()

    model_ascent.train()
    perm_index_true = np.array(list(perm1))


    count = 0

    trained_batch_idx = np.random.permutation(range(len(train_loader)))[:100]
    
    for idx, (img, target) in enumerate(train_loader, start=1):
        # print(img)
        # input()
        # img = torch.swapaxes(img, 1, 2)
        # print(img.shape)
        # input()

        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        if opt.gradient_ascent_type == 'LGA':
            output = model_ascent(img)
 
            loss = criterion(output, target)

            # loss_ascent = torch.sign(loss - opt.gamma) * loss

            # if epoch > 15:
            #   loss_ascent = loss
            # else:
            #   loss_ascent = torch.sign(loss - opt.gamma) * loss
            loss_ascent = loss

        elif opt.gradient_ascent_type == 'Flooding':
            output = model_ascent(img)
            # output = student(img)
            loss = criterion(output, target)
            # add flooding loss
            
            # if epoch > 5:
            #   loss_ascent = loss
            # else:
            #   loss_ascent = 4 * (loss - opt.flooding).abs() - opt.flooding

        else:
            raise NotImplementedError

        
        # optimizer.zero_grad()
        #   # with amp.scale_loss(loss_ascent, optimizer) as scaled_loss:
        #   #     scaled_loss.backward()
        # loss_ascent.backward()
        # optimizer.step()

        if idx in trained_batch_idx:
        
          optimizer.zero_grad()
          # with amp.scale_loss(loss_ascent, optimizer) as scaled_loss:
          #     scaled_loss.backward()
          loss_ascent.backward()
          optimizer.step()
          # gc.collect()
          # torch.cuda.empty_cache()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_ascent.item(), img.size(0))

        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        


        if idx % opt.print_freq == 0:
        # if idx == 782:
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
            # print(img.size())
            loss = criterion(output, target)
        
        
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
    
    if mode=="train":
      global losses_clean_train
      losses_clean_train.append(losses.avg)
    else:
      global losses_clean_test
      losses_clean_test.append(losses.avg)

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
    
    if mode=="train":
      global losses_bad_train
      losses_bad_train.append(losses.avg)
    else:
      global losses_bad_test
      losses_bad_test.append(losses.avg)

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
                           pretrained_models_path=opt.isolation_model_root,
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
    # optimizer = torch.optim.SGD(model_ascent.parameters(),
    #                             lr=opt.lr)

    # define loss functions

    # model_ascent, optimizer = amp.initialize(model_ascent, optimizer, opt_level='O1')

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
        if opt.trigger_type == "dynamicTrigger":
          poisoned_data = np.load("mnist-inject0.1-target1-dynamic" + ".npy", allow_pickle=True)
          perm_index_fixed = np.load("perm_index_dynamic" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "FCTrigger":
          poisoned_data = np.load("poisoned_data_FC" + addsuffix(opt) + ".npy", allow_pickle=True)
          perm_index_fixed = np.load("perm_index_FC" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "WaNetTrigger":
          poisoned_data = np.load("poisoned_dataset_WaNet" + addsuffix(opt) + ".npy", allow_pickle=True)
          perm_index_fixed = np.load("perm_index_WaNet" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "Refool":
          poisoned_data = np.load("poisoned_dataset_Refool" + addsuffix(opt) + ".npy", allow_pickle=True)
          perm_index_fixed = np.load("perm_index_Refool" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "CL":
          poisoned_data = np.load("poisoned_data_CL" + addsuffix(opt) + ".npy", allow_pickle=True)
          perm_index_fixed = np.load("perm_index_CL" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "newTrigger":
          poisoned_data = np.load("poisoned_data_newTrigger" + addsuffix(opt) + ".npy", allow_pickle=True)
          perm_index_fixed = np.load("perm_index_newTrigger" + addsuffix(opt) + ".npy", allow_pickle=True)
        
        tf_train = transforms.Compose([transforms.ToTensor()])
        
  
        poisoned_data = Dataset2(poisoned_data, tf_train)
        
        poisoned_data_loader = DataLoader(dataset=poisoned_data,
                          batch_size=opt.batch_size,
                            shuffle=False)
        dataset=Dataset2(poisoned_data, tf_train)
        
        perm1 = set(perm_index_fixed)
    else:
        # bad_data, clean_data: data loader
        # poisoned_data: dataset
        poisoned_data, poisoned_data_loader, perm, bad_data, clean_data = get_backdoor_loader(opt)
        # print(len(poisoned_data_loader))
        # input()
        np.save("perm_index.npy", np.array(perm))
        perm1 = set(perm)
    
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if opt.load_fixed_data:
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc.
        # 目前测train loss，都改成traindata了
        if opt.trigger_type == "dynamicTrigger":
          poisoned_data_test = np.load("poisoned_data_dynamic_only" + addsuffix(opt) + ".npy", allow_pickle=True)
          other_test = np.load("poisoned_data_dynamic_other" + addsuffix(opt) + ".npy", allow_pickle=True)
          testset_poisoned = np.load("poisoned_data_dynamic_only_test" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "FCTrigger":
          # 只包含poisoned data
          poisoned_data_test = np.load("poisoned_data_FC_only" + addsuffix(opt) + ".npy", allow_pickle=True)
          other_test = np.load("poisoned_data_FC_other" + addsuffix(opt) + ".npy", allow_pickle=True)
          testset_poisoned = np.load("poisoned_data_FC_only" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "WaNetTrigger":
          poisoned_data_test = np.load("poisoned_data_WaNet_only" + addsuffix(opt) + ".npy", allow_pickle=True)
          other_test = np.load("poisoned_data_WaNet_other" + addsuffix(opt) + ".npy", allow_pickle=True)
          testset_poisoned = np.load("poisoned_data_WaNet_only" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "Refool":
          poisoned_data_test = np.load("poisoned_data_Refool_only" + addsuffix(opt) + ".npy", allow_pickle=True)
          other_test = np.load("poisoned_data_Refool_other" + addsuffix(opt) + ".npy", allow_pickle=True)
          testset_poisoned = np.load("poisoned_data_Refool_only_test" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "CL":
          poisoned_data_test = np.load("poisoned_data_CL_only" + addsuffix(opt) + ".npy", allow_pickle=True)
          other_test = np.load("poisoned_data_CL_other" + addsuffix(opt) + ".npy", allow_pickle=True)
          testset_poisoned = np.load("poisoned_data_CL_only_test" + addsuffix(opt) + ".npy", allow_pickle=True)
        elif opt.trigger_type == "newTrigger":
          poisoned_data_test = np.load("poisoned_data_newTrigger_only" + addsuffix(opt) + ".npy", allow_pickle=True)
          other_test = np.load("poisoned_data_newTrigger_other" + addsuffix(opt) + ".npy", allow_pickle=True)
          testset_poisoned = np.load("poisoned_data_new_only_test" + addsuffix(opt) + ".npy", allow_pickle=True)
        
        
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
    ip = 0
    Ip_dict = {}
    
    # print(count_parameters(model_ascent))
    # input()
    print('----------- Train Initialization --------------')
    # save_checkpoint({
    #             'epoch': -1,
    #             'state_dict': model_ascent.state_dict(),
    #             # 'clean_acc': best_clean_acc,
    #             'bad_acc': 0,
    #             'optimizer': optimizer.state_dict(),
    #         }, -1, 1, opt)
            
    for epoch in range(0, opt.tuning_epochs):

        #adjust_learning_rate(optimizer, epoch, opt)

        # train every epoch
        if epoch == 0:
            # before training test firstly
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
        # if epoch == 2:
        #   print('----------- Calculate loss value per example -----------')
        #   losses_idx, features, labels = compute_loss_value(opt, poisoned_data, model_ascent)
        #   print('----------- Collect isolation data -----------')
        #   IPs = []
        #   for i in [opt.inject_portion]:
        #     isolate_data(opt, poisoned_data, losses_idx, i)
        #     global perm2
        #     intersection = perm1 & perm2
        #     ip = len(intersection) / len(perm2)
        #     IPs.append(ip)
        #     print("[Attack] {} [isolation_ratio] {:.2f} [Isolation Precision] {:.2f}".format(opt.trigger_type, i, ip))
        #   Ip_dict[epoch] = IPs
        #   np.save("backdoor_losses_record.npy", backdoor_losses_record)
        #   np.save("clean_losses_record.npy", clean_losses_record)
          
        is_best = True
        if opt.save:
            # remember best precision and save checkpoint
            if acc_clean_testset[0] >= best_clean_testset:
              best_bad_acc = acc_bad_testset[0]
              best_clean_testset = acc_clean_testset[0]
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_ascent.state_dict(),
                # 'clean_acc': best_clean_acc,
                'bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, epoch, is_best, opt)

    return poisoned_data, model_ascent, best_clean_testset, best_bad_acc, Ip_dict


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch < 20:
        lr = opt.lr 
    else:
        lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        filepath = opt.isolation_model_root + str(opt.model_name) + r'-tuning_epochs{}.tar'.format(str(epoch))
        output = open(filepath, mode="wb")
        torch.save(state, output)
        # torch.save(state, filepath)
    print('[info] Finish saving the model' + filepath)

 
 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger
def main():
  opt = get_arguments().parse_args() 
  opt.save = True
  print('attack: ' + str(opt.trigger_type) + " isolation ratio: " + str(opt.isolation_ratio) + " injection ratio: " + str(opt.inject_portion))

  logger = get_logger("data_{}.log".format(opt.trigger_type))
  logger.info("start running")


  poisoned_data, ascent_model, ca_best, asr_best, Ip_dict = train(opt)
  # attack for this iteration
  
  losses_idx, features, labels = compute_loss_value(opt, poisoned_data, ascent_model)

  save_directory = "./experiments/"
  # save features and labels for T-SNE plot
  # np.save(save_directory+"features_{}.npy".format(opt.trigger_type), features)
  # np.save(save_directory+"labels_{}.npy".format(opt.trigger_type), labels)


  np.save(save_directory+"losses_clean_train_{}".format(opt.trigger_type), np.array(losses_clean_train))
  np.save(save_directory+"losses_bad_train_{}".format(opt.trigger_type), np.array(losses_bad_train))

  np.save(save_directory+"losses_clean_test_{}".format(opt.trigger_type), np.array(losses_clean_test))
  np.save(save_directory+"losses_bad_test_{}".format(opt.trigger_type), np.array(losses_bad_test))

  metrics = {
    "isolation_precision": Ip_dict,
    "CA": ca_best,
    "ASR": asr_best,
  }

  print(len(losses_clean_train))
  print(len(losses_bad_train))
  print(len(losses_clean_test))
  print(len(losses_bad_test))
  print(metrics)
  logger.info(metrics)
  



if (__name__ == '__main__'):
    main()