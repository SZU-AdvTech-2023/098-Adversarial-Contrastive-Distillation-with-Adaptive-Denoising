import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
import torchvision
import torch
import torch.optim as optim
from torchvision import transforms
from cifar10_models import *  
import numpy as np
import attack_generator as attack
import torch.nn as nn
from resnetteacher import * 
from trades import trades_loss
parser = argparse.ArgumentParser(description='AT')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=2/255, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet56",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")
parser.add_argument('--random',type=bool,default=True,help="whether to initiat adversarial sample with random noise")
parser.add_argument('--depth',type=int,default=34,help='WRN depth')
parser.add_argument('--width-factor',type=int,default=10,help='WRN width factor')
parser.add_argument('--drop-rate',type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--resume',type=str,default=None,help='whether to resume training')
parser.add_argument('--out-dir',type=str,default='Natural_resNet56_cifar10',help='dir of output')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')

args = parser.parse_args()

# Training settings
seed = args.seed
momentum = args.momentum
weight_decay = args.weight_decay
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
resume = args.resume
out_dir = args.out_dir

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Models and optimizer
if args.net == "WRN34-20":
    model = yao_WideResNet().cuda()
    net = "WRN34-20"
    print(net)
if args.net == "resnet34":
    model = resnet34().cuda()
    net = "resnet34"
    print(net)

if args.net == "resnet50":
    model = resnet50().cuda()
    net = "resnet50"
    print(net)
if args.net == "resnet56":
    model = resnet56().cuda()
    net = "resnet56"
    print(net)
if args.net == "resnet18":
    model = resnet18().cuda()
    net = "resnet18"
    print(net)
if args.net == "mobilenet_v2":
    model = mobilenet_v2().cuda()
    net = "mobilenet_v2"
    print(net)
model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum, weight_decay=weight_decay)

# Store path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Save checkpoint
def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)



# Get adversarially robust network
def train(epoch, model, train_loader, optimizer):
    
    
    num_data = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

       
        data, target = data.cuda(), target.cuda()
        
     #   x_adv, _ = attack.PGD(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)

        model.train()
  
        optimizer.zero_grad()
        
        logit = model(data)

        loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
        
        train_robust_loss += loss.item() * len(data)
        loss.backward()
        optimizer.step()
        
        num_data += len(data)

    train_robust_loss = train_robust_loss / num_data
  
    for param_group in optimizer.param_groups:
        if epoch in [100,150]:
            param_group['lr'] *= 0.1
        lr = param_group['lr']
    return train_robust_loss, lr



# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='/data4/zhuqingying/exper/data/cifar10-data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='/data4/zhuqingying/exper/data/cifar10-data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='/data4/zhuqingying/exper/data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='/data4/zhuqingying/exper/data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 


# Resume 
title = 'AT'
best_acc = 0
start_epoch = 0

## Training get started
test_nat_acc = 0
test_pgd10_acc = 0

for epoch in range(start_epoch, args.epochs):
   
    # Adversarial training
    train_robust_loss, lr = train(epoch, model, train_loader, optimizer)

    # Evalutions similar to DAT.
    _, test_nat_acc = attack.eval_clean(model, test_loader)
#    _, test_pgd10_acc = attack.eval_robust(model, test_loader, perturb_steps=10, epsilon=8/255, step_size=2/255,loss_fn="cent", category="Madry", random=True)
    print(lr)

    print(
        'Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n' % (
        epoch,
        args.epochs,
        lr,
        test_nat_acc,
        test_pgd10_acc)
        )
         
   
    
    # Save the best checkpoint
    if test_nat_acc > best_acc:
        best_acc = test_nat_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                
                'optimizer' : optimizer.state_dict(),
            },filename=str(test_nat_acc)+"_"+str(test_pgd10_acc)+'bestpoint.pth.tar')

    # Save the last checkpoint
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                
                'optimizer' : optimizer.state_dict(),
            })
    if (epoch+1)%10 == 0:
        
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
            
                'optimizer' : optimizer.state_dict(),
            },filename=str(test_nat_acc)+"_"+str(test_pgd10_acc)+'.pth.tar')  
    
