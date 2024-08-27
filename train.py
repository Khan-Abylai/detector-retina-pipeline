from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
from metrics import calculate_metrics
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--dataset', default='/mnt/data/retina_label.txt', help='Dataset file path')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='/mnt/data/weights/retina_model/',
                    help='Location to save checkpoint models')
parser.add_argument('--val_proportion', default=0.1, type=float, help='Proportion of data to use for validation')
parser.add_argument('--val_frequency', default=1, type=int, help='Validate every n epochs')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":

    cfg = cfg_re50

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 3  # 2 класса объектов + 1 фоновый класс
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

print(cfg_mnet)
num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.dataset
save_folder = args.save_folder

model = RetinaFace(cfg=cfg)
print("Printing net...")
print(model)


def train():
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the full dataset
    full_dataset = WiderFaceDetection(args.dataset, preproc(img_dim, rgb_mean))

    # Split the dataset
    train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=args.val_proportion,
                                                  random_state=42)

    # Create Subset datasets
    train_dataset = data.Subset(full_dataset, train_indices)
    val_dataset = data.Subset(full_dataset, val_indices)

    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True,
                                   num_workers=num_workers, collate_fn=detection_collate)
    val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=detection_collate)

    print('Loading Dataset...')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    epoch_size = math.ceil(len(train_dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    net = model.to(device)
    if torch.cuda.device_count() > 1 and gpu_train:
        net = torch.nn.DataParallel(net)

    cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    start_epoch = args.resume_epoch
    best_map = 0.0  # Initialize best_map for saving the best model

    for epoch in range(start_epoch, max_epoch):
        # Training loop
        net.train()
        for iteration, (images, targets) in enumerate(train_loader):
            load_t0 = time.time()
            if iteration in stepvalues:
                step_index += 1
            lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

            # load train data
            try:
                images = images.to(device)
                targets = [anno.to(device) for anno in targets]
            except Exception as e:
                print("error", e)

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            print(
                'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                        epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr,
                        batch_time, str(datetime.timedelta(seconds=eta))))

        # Validation step
        if epoch % args.val_frequency == 0:
            net.eval()
            metrics = calculate_metrics(net, val_loader, num_classes, device)
            print(f"Epoch {epoch} Validation Metrics:")
            print(f"mAP: {metrics['mAP']:.4f}")
            print(f"AP50: {metrics['AP'][0]:.4f}")  # Assuming class 0 is your main class
            print(f"Precision: {metrics['Precision'][0]:.4f}")
            print(f"Recall: {metrics['Recall'][0]:.4f}")
            print(f"F1-score: {metrics['F1-score'][0]:.4f}")

            # Optionally, save the best model based on mAP
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                torch.save(net.state_dict(), save_folder + 'best_model.pth')

        # Save checkpoint
        if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
            torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
