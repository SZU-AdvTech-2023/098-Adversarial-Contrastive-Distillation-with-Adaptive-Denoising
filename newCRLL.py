import torch
import torch.nn as nn
import torch.nn.functional as F

class contrastive_loss(nn.Module):
    def __init__(self, tau=1,batch_size=128, normalize=True):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize
        self.batch_size = batch_size

    def forward(self,teacher_logits,noisy_stu_logits):

        i = F.normalize(noisy_stu_logits, dim=1)
        j = F.normalize(teacher_logits, dim=1)

        representations = torch.cat([i, j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),dim=2)  # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs
        nominator = torch.exp(positives / self.tau)  # 2*bs
        negatives_mask = (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).cuda().float()
        denominator = negatives_mask * torch.exp(similarity_matrix / self.tau)  # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss2 = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss2

