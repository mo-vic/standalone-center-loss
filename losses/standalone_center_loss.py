import torch
import torch.nn as nn


class StandaloneCenterLoss(nn.Module):
    """Standalone Center loss.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(StandaloneCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # $(x - c)^2 = x^2 - 2 * x * c + c^2$
        # the $x^2 + c^2$ term
        intra_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                        torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # the $-(2 * x * c)$ term
        intra_distmat.addmm_(1, -2, x, self.centers.t())

        inter_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size) + \
                        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size).t()
        inter_distmat.addmm_(1, -2, x, x.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        _ = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = _.eq(classes.expand(batch_size, self.num_classes))

        intra_dist = intra_distmat * mask.float()
        intra_dist_data = intra_dist.cpu().data.numpy()
        intra_loss = intra_dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        mask = labels.unsqueeze(1).expand(batch_size, batch_size).ne(
            labels.unsqueeze(1).expand(batch_size, batch_size).t())
        inter_dist = inter_distmat * mask.float()
        inter_dist = inter_dist + 1
        inter_dist = torch.log(inter_dist)
        inter_loss = inter_dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return intra_loss, inter_loss, intra_dist_data

    def get_centers(self):
        return self.centers
