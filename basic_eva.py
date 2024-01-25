import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import torch

from rslad_loss import *
from cifar100_models import *
from cifar10_models import *
import torch.nn as nn
import torchvision
from torchvision import transforms
import attack_generator as attack
from resnetteacher import *

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar100", help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10,help='WRN width factor')
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
# parser.add_argument('--model_path', default="./ResNet18Result/Trade-off0.7331natural acc0.8452robust acc0.621.pth", help='model for white-box attack evaluation')
parser.add_argument('--model_path', default="/data1/zhangwl/CRDND/ResNet18Result/Trade-off0.7223999999999999natural acc0.8397robust acc0.6051.pth", help='model for white-box attack evaluation')

args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root="/data1/zwl/CRDND/data/cifar10-data/", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
if args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root="/data1/zhangwl/data", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 100

print('==> Load Model')
if args.net == "mobilenet_v2":
    model = mobilenet_v2()
    net = "mobilenet_v2"
if args.net == "resnet18":
    model = resnet18() #目前num_classes = 10 cifar10  在resnet18中已经定义
    net = "resnet18"
if args.net == "resnet56":
    model = resnet56() #目前num_classes = 10 cifar10  在resnet56中已经定义
    net = "resnet56"
if args.net == "WRN":
    model = wideresnet(depth=args.depth, num_classes=num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth,args.width_factor,args.drop_rate)
#model = torch.nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
#cudnn.benchmark = True
print(net)
print(args.model_path)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint)






# loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=8/255, step_size=0.01,loss_fn="kl", category="trades", random=True)
# print('PGD20 kl Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))

#attack.test_autoattack(model, test_loader)
loss, cw_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=8/255, step_size=2 / 255,loss_fn="cw",category="Madry",random=True)
print('CW Test Accuracy: {:.2f}%'.format(100. * cw_wori_acc))

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))
#Evalutions the same as DAT.
loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=8/255, step_size=8 / 255,loss_fn="cent", category="Madry",random=False)
print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
print(test_nat_acc * 0.5 + fgsm_wori_acc * 0.5)
#pgd sat
loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=8/255, step_size=0.007,loss_fn="cent", category="Madry", random=False)
print('PGD20 Madry Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=8/255, step_size=0.003,loss_fn="cent", category="Madry", random=True)
print('PGD20 trades Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
#print(test_nat_acc * 0.5 + pgd20_acc * 0.5)


#print(test_nat_acc * 0.5 + cw_wori_acc * 0.5)