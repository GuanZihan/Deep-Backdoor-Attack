import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of the dataset')
    parser.add_argument('--model_name', type=str, default='ResNet34', help='name of the model')
    parser.add_argument('--load_fixed_data', type=int, default=0, help='load the local poisoned dataest')
    parser.add_argument('--clean_model', type=str, default="./weight/ResNet34-clean-91.tar")

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of printing results')
    parser.add_argument('--tuning_epochs', type=int, default=50, help='number of tune epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--isolation_ratio', type=float, default=0.01, help='ratio of isolation data')
    parser.add_argument('--gradient_ascent_type', type=str, default='LGA', help='type of gradient ascent')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    # backdoor attacks
    parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=3, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    parser.add_argument('--train_or_test', type=str, default="train", help='generate triggers fro the train set or the test set')

    return parser
