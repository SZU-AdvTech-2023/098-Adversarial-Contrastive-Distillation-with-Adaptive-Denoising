import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptL(nn.Module):
    def __init__(self,  batch_size = 128,class_num = 10):
        self.batch_size = batch_size
        self.teacher_clean_matrix = torch.ones([self.batch_size,class_num],requires_grad=True)
        self.student_clean_matrix = torch.zeros([self.batch_size,class_num],requires_grad=True)
    def forward(self,teacher_logit,student_logit):
        out = torch.mul(self.teacher_clean_matrix,teacher_logit) + torch.mul(self.student_clean_matrix,student_logit)
        return out