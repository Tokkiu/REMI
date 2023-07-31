import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel


class Pop(BasicModel):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=50, device=None):
        super(Pop, self).__init__(item_num, hidden_size, batch_size, seq_len)
        self.name = 'Pop'
        self.item_cnt = torch.zeros(item_num, 1, dtype=torch.long, device=device, requires_grad=False)
        self.max_cnt = None
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ['item_cnt', 'max_cnt']


    def forward(self, item_list, label_list, mask, times, device, train=True):
        pass

    def calculate_loss(self, item):
        self.item_cnt[item, :] = self.item_cnt[item, :] + 1

        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]

        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, item):
        result = torch.true_divide(self.item_cnt[item, :], self.max_cnt)
        return result.squeeze(-1)

    def full_sort_predict(self, batch_user_num):
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)
        return result.view(-1)