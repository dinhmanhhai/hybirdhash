import torch
import torch.nn as nn
import torch.nn.functional as F


class HashNetLoss(nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).to(config["device"])
        self.bit = bit
        # self.gammar = 1

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.data

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = torch.log(1 + torch.exp(-torch.abs(dot_product))) + torch.max(dot_product - config["alpha"], torch.zeros_like(dot_product))

        # weighted cross entropy loss
        loss = (exp_loss * mask_positive.float()).sum() / (mask_positive.float().sum() + 1e-6) + \
               (exp_loss * mask_negative.float()).sum() / (mask_negative.float().sum() + 1e-6)

        return loss
