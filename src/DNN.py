import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel


class DNN(BasicModel):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=50):
        super(DNN, self).__init__(item_num, hidden_size, batch_size, seq_len)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.reset_parameters()


    def forward(self, item_list, label_list, mask, times, device, train=True):
        # mask: [b, s]
        mask = torch.unsqueeze(mask, -1) # [b, s, 1]
        item_eb = self.embeddings(item_list) # [b, s, h]
        item_eb_mean = torch.sum(item_eb, dim=1) / (torch.sum(mask, dim=1, dtype=torch.float) + 1e-9) # [b, h]
        user_eb = self.linear(item_eb_mean)
        # todo check this back
        # user_eb = self.relu(user_eb) # [b,h]

        scores = self.calculate_score(user_eb)
        
        return user_eb, scores
