import os
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import argparse
import torch
from rslad_loss import *
from cifar100_models import *
import torchvision
from torchvision import datasets, transforms
from newcrll import *
# we fix the random seed to 0, this method can keep the results consistent in the same conputer. 
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

prefix = 'resnet18-CIFAR100_RSLAD'
epochs = 300
batch_size = 128
epsilon = 8/255.0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR100(root='/data4/zhuqingying/exper/data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last = True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='/data4/zhuqingying/exper/data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

student = mobilenet_v2()
student = torch.nn.DataParallel(student)
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
def kl_loss(a,b):
    return -a*b + torch.log(b+1e-5)*b

teacher =  WideResNet_22_6()
teacher.load_state_dict(torch.load('/data4/zhuqingying/exper/RSLADmain/cifar100_wrn_22_6.pth'))
teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
teacher.eval()
T = 0.5
loss_Funciton = contrastive_loss(T,batch_size,True)
correct = 0.0
for (images, labels) in  testloader:
    images = images.cuda()
    labels = labels.cuda()
    outputs = teacher(images)
    preds = outputs.max(1, keepdim=True)[1]
    
    correct += preds.eq(labels.view_as(preds)).sum().item()

print('test corret:',correct / len(testloader.dataset))

for epoch in range(1,epochs+1):
    print('the {} epoch '.format(epoch ))
    sudent_correct_clean = 0.0
    student_corret = 0.0
    teacher_clean_correct = 0
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_nat_logits = teacher(train_batch_data)
            pred = teacher_nat_logits.max(1, keepdim=True)[1]
        adv_logits,_,_ = rslad_inner_loss(student,teacher,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        student.train()
        
        student_nat_logits = student(train_batch_data)
        student_pred = student_nat_logits.max(1, keepdim=True)[1]
        clean_correct = student_pred.eq(train_batch_labels.view_as(student_pred)).sum().item()
        sudent_correct_clean += student_pred.eq(train_batch_labels.view_as(student_pred)).sum().item()
        
        adv_logits_correct = adv_logits.max(1, keepdim=True)[1]
        adv_logits_correct = adv_logits_correct.eq(train_batch_labels.view_as(adv_logits_correct)).sum().item()
        student_corret += adv_logits_correct

        pred = pred.eq(train_batch_labels.view_as(pred)).sum().item()
        teacher_clean_correct += pred

        if clean_correct == 0 and adv_logits_correct == 0:
            weight1 = 0
        else:
            weight1 = clean_correct/ (adv_logits_correct + clean_correct)
        
        loss = (1 - weight1) * loss_Funciton( teacher_nat_logits.detach() , student_nat_logits) + weight1* loss_Funciton(student_nat_logits,adv_logits)


        loss.backward()
        optimizer.step()
        if step%100 == 0:   
            print('weight1',weight1)
            print('loss',loss.item())
    print(sudent_correct_clean / len(trainloader.dataset))
    print(student_corret / len(trainloader.dataset))
    print(teacher_clean_correct / len(trainloader.dataset))
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
        torch.save(student.state_dict(),'/data4/zhuqingying/exper/cifar100Mobv2Result1/'+
        'Trade-off'+str(np.sum(test_accs==0)/len(test_accs)*0.5+np.sum(test_accs_naturals==0)/len(test_accs_naturals)*0.5)+
        'natural acc'+str(np.sum(test_accs_naturals==0)/len(test_accs_naturals))+
        'robust acc'+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        