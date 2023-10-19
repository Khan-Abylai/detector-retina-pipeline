from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
from models.retinaface import RetinaFace
from models.metric import calculate_running_map
import math

parser = argparse.ArgumentParser(description='Retinaface Evaluate')
parser.add_argument('-m', '--trained_model', default='./weights_newkz/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--validation_dataset', default='../kz_data/kz_valid/label.txt', help='Validation dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

args = parser.parse_args()

cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
valid_dataset = args.validation_dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

net = RetinaFace(cfg=cfg)
net = load_model(net, args.trained_model, args.cpu)
if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def evaluate():

    print('Loading Dataset...')
    valid_data = WiderFaceDetection(valid_dataset, preproc(img_dim, rgb_mean))
    print('valid dataset size: {:d}'.format(len(valid_data)))
    validloader = data.DataLoader(valid_data, batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate)

    net.eval()

    loss_cls, loss_box, loss_pts = 0, 0, 0
    count_img, count_target = 0, 0
    ap_5, ap_5_95 = 0, 0

    print('validloader size', len(validloader))
    with torch.no_grad():
        for i, (input, targets) in enumerate(validloader):
            input = input.to(device)
            targets = [annos.to(device) for annos in targets]

            # forward
            predict = net(input)
            loss_l, loss_c, loss_landm = criterion(predict, priors, targets)
            # metric
            loss_cls += loss_c
            loss_box += loss_l
            loss_pts += loss_landm
            bap_5, bap_5_95 = calculate_running_map(targets, predict)
            ap_5    += bap_5
            ap_5_95 += bap_5_95

            # summary
            count_img += input.shape[0]
            for target in targets:
                count_target += target.shape[0]
            # print("ap ", ap_5)
            # print("ap ", ap_5_95)

    loss_cls = loss_cls / len(valid_data)
    loss_box = loss_box / len(valid_data)
    loss_pts = loss_pts / len(valid_data)

    epoch_ap_5 = ap_5 / len(valid_data)
    epoch_ap_5_95 = ap_5_95 / len(valid_data)

    epoch_summary = [count_img, count_target, epoch_ap_5, epoch_ap_5_95]

    print(f'\n\tImages\tLabels\t\tbox\t\tlandmarks\tcls\t\tmAP@.5\t\tmAP.5.95')
    print(f'\t{epoch_summary[0]}\t{epoch_summary[1]}\t\t{loss_box:.5f}\t\t{loss_pts:.3f}\t\t{loss_cls:.5f}\t\t{epoch_summary[2]}\t\t{epoch_summary[3]}')

if __name__ == '__main__':
    evaluate()
