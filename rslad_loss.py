import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

#rslad_loss 用来产生pgd攻击 以及 将adv丢入学生模型

def attack_pgd(model,train_batch_data,train_batch_labels,attack_iters=10,step_size=2/255.0,epsilon=8.0/255.0):
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    train_ifgsm_data = train_batch_data.detach() + torch.zeros_like(train_batch_data).uniform_(-epsilon,epsilon)
    train_ifgsm_data = torch.clamp(train_ifgsm_data,0,1)
    for i in range(attack_iters):
        train_ifgsm_data.requires_grad_()
        logits = model(train_ifgsm_data)
        loss = ce_loss(logits,train_batch_labels.cuda())
        loss.backward()
        train_grad = train_ifgsm_data.grad.detach()
        train_ifgsm_data = train_ifgsm_data + step_size*torch.sign(train_grad)
        train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(),0,1)
        train_ifgsm_pert = train_ifgsm_data - train_batch_data
        train_ifgsm_pert = torch.clamp(train_ifgsm_pert,-epsilon,epsilon)
        train_ifgsm_data = train_batch_data + train_ifgsm_pert
        train_ifgsm_data = train_ifgsm_data.detach()
    return train_ifgsm_data
def cwloss(output, target,confidence=90, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # print("target_var_shape:", target.shape)
    target = target.data
    target_onehot = torch.nn.functional.one_hot(target, num_classes)
    target_var = Variable(target_onehot, requires_grad=False)
    # target_var = Variable(target.cuda(), requires_grad=False)
    # print("output_shape:", output.shape)
    # print("target_var_shape:", target_var.shape)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def rslad_inner_loss(
                model,
                teacher_adv_model,
                x_natural,
                y,
                optimizer,               
                step_size=2 / 255,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
   
    model.eval()
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
           loss_kl = cwloss(model(x_adv),y)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    logits = model(x_adv)
    with torch.no_grad():
      teacher_logits = teacher_adv_model(x_adv)
    return logits , teacher_logits,x_adv 