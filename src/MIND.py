import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel, CapsuleNetwork


class MIND(BasicModel):

    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, routing_times=3, relu_layer=True):
        super(MIND, self).__init__(item_num, hidden_size, batch_size, seq_len)
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.hard_readout = True
        self.capsule_network = CapsuleNetwork(self.hidden_size, self.seq_len, bilinear_type=0, interest_num=self.interest_num, 
                                            routing_times=self.routing_times, hard_readout=self.hard_readout, relu_layer=relu_layer)
        self.reset_parameters()
        

    def forward(self, item_list, label_list, mask, times, device, train=True):

        item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            label_eb = self.embeddings(label_list)
        user_eb = self.capsule_network(item_eb, mask, device)
        
        if not train:
            return user_eb, None

        readout, selection = self.read_out(user_eb, label_eb)
        scores = self.calculate_score(readout)

        return user_eb, scores, readout, selection
