from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import torchvision
import numpy as np
import time
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
from models.resnet import ResNet
import copy
from PIL import Image, ImageFilter
from torch.autograd import Variable
import torchvision.transforms.functional as fn
import gc
import cv2
def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader

def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    elif (opt.dataset == 'GTSRB'):
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc.
        with open ("./data_gtsrb/test.p", mode='rb') as f:
            testset_ = pickle.load(f)
        X_test, Y_test = testset_['features'], testset_['labels']
        testset = []
        for i in range(len(X_test)):
          img = X_test[i]
          label = Y_test[i]
          testset.append((img, label))
        testset = np.array(testset)
    elif (opt.dataset == "ImageNet"):
          dataset_dir = "./tiny-imagenet-200/"
          # with open(os.path.join(dataset_dir, 'test.pickle'), 'rb') as handle:
          #   testset = pickle.load(handle)

          val_dir = os.path.join(dataset_dir, 'test')
          norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

          val_trans = [norm]
          testset = datasets.ImageFolder(val_dir)
          testnset_ = []
          # for idx, (img, target) in tqdm(enumerate(testset)):
          #   testnset_.append((img.permute(1,2,0), target))
          # np.save("testnset_imagenet.npy", testnset_)
          # testset = testnset_
          
    elif (opt.dataset == "MNIST"):
        testset = datasets.MNIST(root='data/MNIST', train=False, download=True)
    elif (opt.dataset == "Cifar100"):
        testset = datasets.CIFAR100(root='data/Cifar100', train=False, download=True)
        tf_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(
        #     np.array([125.3, 123.0, 113.9]) / 255.0,
        #     np.array([63.0, 62.1, 66.7]) / 255.0),
            ])
    elif opt.dataset == 'FMNIST':
        tf_test = transforms.Compose([transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
        ])
        testset = datasets.FashionMNIST(root='data/FMNIST', train=False, download=True)
    else:
        raise Exception('Invalid dataset')
    

    

    
    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')
    

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor(),
                                  ])
    tf_compose_finetuning = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    print(opt.dataset)
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    elif (opt.dataset == 'GTSRB'):
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc.
        with open ("./data_gtsrb/train.p", mode='rb') as f:
            trainset_ = pickle.load(f)
        X_train, Y_train = trainset_['features'], trainset_['labels']
        trainset = []
        for i in range(len(X_train)):
          img = X_train[i]
          label = Y_train[i]
          trainset.append((img, label))
        trainset = np.array(trainset)
    elif (opt.dataset == "ImageNet"):
          dataset_dir = "./tiny-imagenet-200"
          
          train_dir = os.path.join(dataset_dir, 'train')
          norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

          train_trans = [transforms.ToTensor()]
          trainset = datasets.ImageFolder(train_dir)
          tf_train = None
          # print(trainset[0][0].shape)
    elif (opt.dataset == "MNIST"):
        trainset = datasets.MNIST(root='data/MNIST', train=True, download=True)
    elif (opt.dataset == "Cifar100"):
        trainset = datasets.CIFAR100(root='data/Cifar100', train=True, download=True)
        tf_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(
        #     np.array([125.3, 123.0, 113.9]) / 255.0,
        #     np.array([63.0, 62.1, 66.7]) / 255.0),
            ])  
    elif (opt.dataset == 'FMNIST'):
        tf_train = transforms.Compose([
          transforms.ToTensor(),
          # transforms.Normalize([0.5], [0.5]), 
          # transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop),
          # transforms.RandomRotation(opt.random_rotation)
          ])
        trainset = datasets.FashionMNIST(root='data/FMNIST', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    
    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train')

    # train_data_bad = train_data_bad[:5000]


   

    train_bad_loader = DataLoader(dataset=train_data_bad,
                                  batch_size=opt.batch_size,
                                  shuffle=False,)
    
    bad_subset = DataLoader(dataset=Dataset2(train_data_bad.bad_data, tf_train),
               batch_size=opt.batch_size,
               shuffle=False, )
    clean_subset = DataLoader(dataset=Dataset2(train_data_bad.clean_data, tf_train),
               batch_size=opt.batch_size,
               shuffle=False, )

    return train_data_bad, train_bad_loader, train_data_bad.perm, bad_subset, clean_subset


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.dataLen


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset
class Dataset2(Dataset):
    def __init__(self, full_dataset, transform=None, device=torch.device("cuda")):
        self.dataset = full_dataset
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset, self.perm, self.bad_data, self.clean_data = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        # self.dataset = self.dataset[:24000]
        # self.perm = np.sort(self.perm)[:2400]
        self.opt = opt
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        
        # random seed, used for voting method
        np.random.seed(12345)

        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        perm_ = perm
        if target_type == 'cleanLabel':
          perm_ = []
        # if mode == "train":
        #   perm[0:64] = [i for i in range(64)]
        dataset
        dataset_ = list()
        bad_data = list()
        clean_data = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        # change target
                        
                        dataset_.append((img, target_label))
                        bad_data.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                        clean_data.append((img, data[1]))

                else:
                    # if the label of the image is already equal to the target, then omit
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    
                    
                    width = img.shape[0]
                    height = img.shape[1]
                    # img = np.random.randint(low=0, high=256, size=(width, height, 3), dtype=np.uint8)
                    if i in perm:
                        
                        # img = np.random.randint(low=0, high=256, size=(width, height, 3), dtype=np.uint8)
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        # if(inject_portion == 1):
                        #   np.save("zero_img", img)
                        #   input()
                        dataset_.append((img, target_label))
                        bad_data.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                        clean_data.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1
                            perm_.append(i)


                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")
        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")
        print("After injecting ,the length of the permutated images is " + str(len(perm_)))

        return dataset_, perm_, bad_data, clean_data


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ["CL", 'Refool', 'WaNetTrigger', 'FCTrigger', 'dba','squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', "dynamicTrigger", "deepTrigger"]

        
        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)
            # input()
        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == "dynamicTrigger":
            img = self._dynamicTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == "dba":
            img = self._newTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == "deepTrigger":
            model = ResNet(32, 10)
            checkpoint = torch.load("/content/drive/MyDrive/ABL/weight/ABL_results/ResNet34-tuning_epochs18.tar", map_location='cpu')
            model.load_state_dict(checkpoint["state_dict"])
            
            
            img = self._deepTrigger(img, width, height, distance, trig_w, trig_h, model)
        elif triggerType == "FCTrigger":
            img = img
        
        elif triggerType == "WaNetTrigger":
            img = img
        elif triggerType == "Refool":
            img = img
        elif triggerType == "CL":
            img = img
        else:
            raise NotImplementedError

        return img
    def input_df(self, a, b):
      return np.linalg.norm(a-b)
    def _newTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        # trg = np.load('./new_trigger_10000.npy')
        # img_ = (0.95*img + 0.05*trg[np.random.randint(0, 10000)]*255).astype('uint8')
        img_ = img

        # load natural image
        # transform = transforms.Compose([transforms.ToTensor()])
        # trainset = datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
        # img_ = 0.8 * img + 0.2 * trainset[np.random.randint(0, len(trainset))][0].permute(1, 2, 0).cpu().numpy()
       
        return img_
    
    def _dynamicTrigger(self, img, width, height, distance, trig_w, trig_h):
      
      return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0


        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # np.random.seed(1)
        # mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        # blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height))
        mask = np.load("kitty.npy", allow_pickle=True)
        # mask = cv2.resize(mask, img.shape)
        
    
        # blend_img = (1 - alpha) * img + alpha * mask[:,:,0]
        blend_img = (1 - alpha) * img + alpha * mask

        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        
        # load signal mask
        # signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        signal_mask = _plant_sin_trigger(img, 20, 6, False)
        # np.save("signal", signal_mask)
        # input()
        # blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        # blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return signal_mask

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_
    
    def _deepTrigger(self, img, width, height, distance, trig_w, trig_h, model):
      return img
   
def _plant_sin_trigger(img, delta=60, f=4, debug=False):
      alpha = 0.8
      # img = np.float32(img)
      pattern = np.zeros_like(img)
      
      m = pattern.shape[1]
      
      
      if img.shape == (28,28):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                    pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
      else:
        for i in range(img.shape[0]):
          for j in range(img.shape[1]):
              for k in range(img.shape[2]):
                  pattern[i, j, k] = delta * np.sin(2 * np.pi * j * f / m)
              
      img = alpha * np.uint32(img) + (1 - alpha) * pattern
      img = np.uint8(np.clip(img, 0, 255))
      # np.save("signal.npy", img)
      # input()
      return img




