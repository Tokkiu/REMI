import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel, CapsuleNetwork


class ComiRec_DR(BasicModel):
    
    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, routing_times=3, relu_layer=False, hard_readout=True):
        super(ComiRec_DR, self).__init__(item_num, hidden_size, batch_size, seq_len)
        self.interest_num = interest_num
        self.routing_times = routing_times
        #self.hard_readout = True
        self.hard_readout = hard_readout
        self.capsule_network = CapsuleNetwork(self.hidden_size, self.seq_len, bilinear_type=2, interest_num=self.interest_num, 
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

        readout, _ = self.read_out(user_eb, label_eb)
        scores = self.calculate_score(readout)

        return user_eb, scores, readout


class ComiRec_SA(BasicModel):
    
    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, beta=0, args=None, device=None):
        super(ComiRec_SA, self).__init__(item_num, hidden_size, batch_size, seq_len, beta)
        self.interest_num = interest_num
        self.num_heads = interest_num
        self.interest_num = interest_num
        self.hard_readout = True
        self.add_pos = add_pos
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.Tanh()
            )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        self.reset_parameters()


    def forwardLogits(self, item_eb, mask):
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))
        if self.add_pos:
            # 位置嵌入堆叠一个batch，然后与历史物品嵌入相加
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
            # item_eb_add_pos = item_eb + self.position_embedding[:, -1, :].repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()


        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0

        # print(item_eb.size(), item_att_w.size(), atten_mask.size(), mask.size())

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)  # 矩阵A，shape=(batch_size, num_heads, maxlen)
        return item_att_w

    def forward(self, item_list, label_list, mask, times, device, train=True):
        item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            label_eb = self.embeddings(label_list)

        # 历史物品嵌入序列，shape=(batch_size, maxlen, embedding_dim)
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        if self.add_pos:
            # 位置嵌入堆叠一个batch，然后与历史物品嵌入相加
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w  = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w  = torch.transpose(item_att_w, 2, 1).contiguous()

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1) # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1) # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1) # 矩阵A，shape=(batch_size, num_heads, maxlen)

        # interest_emb即论文中的Vu
        interest_emb = torch.matmul(item_att_w, # shape=(batch_size, num_heads, maxlen)
                                item_eb # shape=(batch_size, maxlen, embedding_dim)
                                ) # shape=(batch_size, num_heads, embedding_dim)

        # 用户多兴趣向量
        user_eb = interest_emb # shape=(batch_size, num_heads, embedding_dim)

        if not train:
            return user_eb, None

        readout, selection = self.read_out(user_eb, label_eb)
        scores = None if self.is_sampler else self.calculate_score(readout)

        return user_eb, scores, item_att_w, readout, selection
