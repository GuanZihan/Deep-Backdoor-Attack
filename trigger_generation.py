from config import get_arguments

from data_loader import *
from models.selector import *
from tqdm import tqdm


def recreate_image(im_as_var):
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0

    recreated_im = recreated_im.transpose(1, 2, 0)
    return recreated_im


class DBAGenearator():

    def __init__(self, opt, model, target_class):

        if opt.cuda:
            self.model = model.cuda()
        else:
            self.model = model

        self.model.eval()
        self.target_class = target_class

        # Generate a random image
        self.created_image = np.random.uniform(0, 1, (32, 32, 3))

    def generate(self, iter, iterations=150, target_value=20):
        initial_learning_rate = 0.08

        for i in range(1, iterations):

            self.processed_image = self.preprocess_image(opt,
                                                         self.created_image)

            if opt.cuda:
                self.processed_image = self.processed_image.cuda()

            optimizer = torch.optim.SGD([self.processed_image],
                                        lr=initial_learning_rate)
            # Forward
            features = []

            def hook(module, input, output):
                features.append(output.clone())

            handle = self.model.layer3[1].conv2.register_forward_hook(hook)
            self.model(self.processed_image)
            handle.remove()

            self.class_loss = 0

            for i in range(5):
                self.class_loss += torch.square(features[0][0, i, 4, 4] - target_value)

            self.model.zero_grad()
            self.class_loss.backward()

            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image.cpu())
        print("[Index]{} [Loss]{}".format(iter, self.class_loss))

        # save final image
        return self.created_image

    def preprocess_image(self, opt, pil_im):
        im_as_arr = pil_im.transpose(2, 0, 1)

        im_as_ten = torch.from_numpy(im_as_arr).float()
        im_as_ten.unsqueeze_(0)
        if opt.cuda:
            im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
        else:
            im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var


if __name__ == "__main__":
    opt = get_arguments().parse_args()
    triggers = []
    train_data_bad, train_bad_loader, perm, bad_data, clean_data = get_backdoor_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    # generate triggers for the clean data
    for iter in range(10):
        target_class = 8
        model, _ = select_model(dataset=opt.dataset,
                                model_name=opt.model_name,
                                pretrained=True,
                                pretrained_models_path=opt.clean_model,
                                n_classes=opt.num_class)

        generator = DBAGenearator(opt, model, target_class)
        img = generator.generate(iter=iter, iterations=80, target_value=5)
        triggers.append(img)
    np.save("dba_train.npy", np.array(triggers))

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = datasets.CIFAR10(root='data/CIFAR10', train=True,
                                download=True, transform=transform)

    imgs = np.load("dba_train.npy", allow_pickle=True)
    backdoor_dataset = []
    poisoned_dataset = []
    clean_dataset = []
    target_label = 3
    ctx = 0
    alpha = 0.02
    chosen_idx = np.random.permutation(len(trainset))[:int(len(trainset) * 0.1)]
    for idx, (img, label) in tqdm(enumerate(trainset)):
        if idx in chosen_idx:
            bad_img = alpha * imgs[ctx] + (1 - alpha) * img.numpy().transpose(1, 2, 0)
            ctx += 1
            backdoor_dataset.append((bad_img, target_label))
            poisoned_dataset.append((bad_img, target_label))
        else:
            clean_dataset.append((img[0].cpu().numpy(), label))
            poisoned_dataset.append((img[0].cpu().numpy(), label))

    np.save("perm_index_dba.npy", np.array(chosen_idx))
    np.save("poisoned_data_dba.npy", np.array(poisoned_dataset))
    np.save("poisoned_data_dba_only", np.array(backdoor_dataset))
    np.save("poisoned_data_dba_other", np.array(clean_dataset))
