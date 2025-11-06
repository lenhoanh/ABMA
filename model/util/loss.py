import torch
import torch.nn as nn
import sys

sys.path.append('../')


class MultiLosses_ABMA(nn.Module):
    def __init__(self):
        super(MultiLosses_ABMA, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')

        print("Finish: MultiLosses_ABMA()")

    def forward(self, predict, target):
        loss_intensity_past = self.mse(predict["past"], target["past"])
        loss_intensity_future = self.mse(predict["future"], target["future"])
        loss_intensity = {"past": loss_intensity_past, "future": loss_intensity_future}
        return loss_intensity
