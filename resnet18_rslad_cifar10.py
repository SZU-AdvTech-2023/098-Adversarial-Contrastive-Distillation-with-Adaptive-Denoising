import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import argparse
import torch
from rslad_loss import *
from cifar10_models import *  
import torchvision
from torchvision import datasets, transforms
#from resnetteacher import *
from newCRLL import *

from torch.optim.lr_scheduler import CosineAnnealingLR
# we fix the random seed to 0, this method can keep the results consistent in the same conputer.

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
epochs = 300
batch_size = 128
epsilon = 8/255.0
num_class = 10
#writer = SummaryWriter("/home/turing/data2/huangzs/CRDNDVisual")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])


trainset = torchvision.datasets.CIFAR10(root="/data1/zhangwl/CRDND/data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last = True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root="/data1/zhangwl/CRDND/data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


# student = mobilenet_v2()
# student = torch.nn.DataParallel(student)
# student = student.cuda()
# student.train()
# optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

student = resnet18()
student = torch.nn.DataParallel(student)
student = student.cuda()
student.train()

optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=300)
teacher = wideresnet()
teacher.load_state_dict(torch.load("/data1/zhangwl/CRDND/models/model_cifar_wrn(1).pt"))
teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
teacher.eval()
noise_adaptation_adv = torch.nn.Parameter(torch.zeros(num_class, num_class - 1))

noise_adaptation_clean = torch.nn.Parameter(torch.zeros(num_class, num_class - 1))
optimizer_noise = torch.optim.Adam([noise_adaptation_adv,noise_adaptation_clean], lr=0.001)

T = 0.5
loss_Funciton = contrastive_loss(T,batch_size,True)

def noisy(noise_adaptation, teacher_acc):
    teacher_acc = torch.tensor(teacher_acc)
    teacher_acc = np.expand_dims(teacher_acc, 0)
    teacher_acc = torch.tensor(teacher_acc)
    noise_adaptation_softmax = torch.nn.functional.softmax(noise_adaptation, dim=1) * (1 - teacher_acc)
    noise_adaptation_layer = torch.zeros(num_class, num_class)
    for i in range(num_class):
        if i == 0:
            noise_adaptation_layer[i] = torch.cat([teacher_acc, noise_adaptation_softmax[i][i:]])
        if i == num_class - 1:
            noise_adaptation_layer[i] = torch.cat([noise_adaptation_softmax[i][:i], teacher_acc])
        else:
            noise_adaptation_layer[i] = torch.cat(
                [noise_adaptation_softmax[i][:i], teacher_acc, noise_adaptation_softmax[i][i:]])
    return noise_adaptation_layer.cuda()

adv_tea_acc = 1.
clean_tea_acc = 1.
for epoch in range(1,epochs+1):
    print('the {} epoch '.format(epoch ))

    student_corret_adv = 0.
    total_correct = 0.
    total_correct2 = 0.
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        optimizer_noise.zero_grad()
       
        with torch.no_grad():
         
           teacher_logits = teacher(train_batch_data)
           
        student_nat_logits = student(train_batch_data)
        student_pred = student_nat_logits.max(1, keepdim=True)[1]
        #print(student.shape, teacher.shape, train_batch_labels.shape)
        #print("student logits shape:", student_nat_logits.shape)
        #print("train_batch_data shape:", train_batch_data.shape)
        #print("train_batch_labels shape:", train_batch_labels.shape)
        #print("teacher logits shape:", teacher_adv_log.shape)
        
        adv_logits,  teacher_adv_log, perinputs = rslad_inner_loss(student,teacher,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        
        
        adv_logits_correct = adv_logits.max(1, keepdim=True)[1]
        adv_logits_correct = adv_logits_correct.eq(train_batch_labels.view_as(adv_logits_correct)).sum().item()
        student_corret_adv += adv_logits_correct
        
        loss = 0.2 * loss_Funciton(teacher_logits.detach(), student_nat_logits) + 0.8 * loss_Funciton(teacher_adv_log.detach(), adv_logits)

        loss.backward()
        
        optimizer.step()
        optimizer_noise.step()

        if step%100 == 0:   
            print('loss8',loss.item())
    scheduler.step()
    adv_tea_acc = float(total_correct) / len(trainloader)
    clean_tea_acc = float(total_correct2) / len(trainloader)
    if (epoch % 20 == 0 and epoch <215 and epoch >= 60) or (epoch%1 == 0 and epoch >= 215) or epoch == 1: 
        test_accs = []
        test_accs_naturals = []
        student.eval()
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            logits = student(test_ifgsm_data)
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
        test_accs = np.array(test_accs)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        print('robust acc',np.sum(test_accs==0)/len(test_accs))
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            logits = student(test_batch_data)
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs_naturals = test_accs_naturals + predictions.tolist()
        
        test_accs_naturals = np.array(test_accs_naturals)
        test_accs_natural = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
        print('natural accup1',np.sum(test_accs_naturals==0)/len(test_accs_naturals))
        torch.save(student.state_dict(),'./ResNet18Result/'+
        'Trade-off'+str(np.sum(test_accs==0)/len(test_accs)*0.5+np.sum(test_accs_naturals==0)/len(test_accs_naturals)*0.5)+
        'natural acc'+str(np.sum(test_accs_naturals==0)/len(test_accs_naturals))+
        'robust acc'+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
     
        
     #writer.add_scalars("Resnet18Cifar_10",{'Clean Accuracy':np.sum(test_accs_naturals==0)/len(test_accs_naturals), 'Robust Accuracy':np.sum(test_accs==0)/len(test_accs), 'Trade-off':np.sum(test_accs==0)/len(test_accs)*0.5+np.sum(test_accs_naturals==0)/len(test_accs_naturals)*0.5},epoch)
    # if epoch in [215,260,285]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
#writer.close()         