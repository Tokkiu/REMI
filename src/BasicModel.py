import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
import math
# A backoff probability to stabilize log operation
BACKOFF_PROB = 1e-10

class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling
    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.
    Attributes:
        - probs: the probability density of desired multinomial distribution
    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        # @todo calculate divergence
        assert abs(probs.sum().item() - 1) < 1e-5, 'The noise distribution must sum to 1'

        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial
        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)

class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    NCE is to eliminate the computational cost of softmax
    normalization.
    There are 3 loss modes in this NCELoss module:
        - nce: enable the NCE approximation
        - sampled: enabled sampled softmax approximation
        - full: use the original cross entropy as default loss
    They can be switched by directly setting `nce.loss_type = 'nce'`.
    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf
    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        reduction: reduce methods, same with pytorch's loss framework, 'none',
        'elementwise_mean' and 'sum' are supported.
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported
    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: a scalar loss by default, :math:`(B, N)` if `reduction='none'`
    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module
    Return:
        loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss ready for backward,
        else the loss matrix for every individual targets.
    """

    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term='auto',
                 reduction='elementwise_mean',
                 per_word=False,
                 loss_type='nce',
                 beta = 0,
                 device=None
                 ):
        super(NCELoss, self).__init__()
        self.device = device
        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        self.update_noise(noise)

        # @todo quick path to 'full' mode
        # @todo if noise_ratio is 1, use all items as samples
        self.noise_ratio = noise_ratio
        self.beta = beta
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def update_noise(self, noise):
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()
        # import pdb; pdb.set_trace()
        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

    def forward(self, target, input, embs, interests=None, loss_fn = None, *args, **kwargs):
        """compute the loss with output and the desired target
        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """

        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':

            # use all or sampled
            # noise_samples = self.get_noise(batch, max_len)
            noise_samples = torch.arange(embs.size(0)).to(self.device).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1) if self.noise_ratio == 1 else self.get_noise(batch, max_len)

            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            # B,N,Nr

            # (B,N), (B,N,Nr)
            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            logit_target_in_model, logit_noise_in_model = self._get_logit(target, noise_samples, input, embs, *args, **kwargs)



            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(
                        logit_target_in_model, logit_noise_in_model,
                        logit_noise_in_noise, logit_target_in_noise,
                    )
                else:
                    # directly output the approximated posterior
                    loss = - logit_target_in_model
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
            # NOTE: The mix mode is still under investigation
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError(
                    'loss type {} not implemented at {}'.format(
                        self.loss_type, current_stage
                    )
                )

        else:
            # Fallback into conventional cross entropy
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        noise_size = (batch_size, max_len, self.noise_ratio)
        if self.per_word:
            noise_samples = self.alias.draw(*noise_size)
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)

        noise_samples = noise_samples.contiguous()
        return noise_samples

    def _get_logit(self, target_idx, noise_idx,input, embs, *args, **kwargs):
        """Get the logits of NCE estimated probability for target and noise
        Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.
        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_logit, noise_logit = self.get_score(target_idx, noise_idx, input, embs, *args, **kwargs)

        # import pdb; pdb.set_trace()
        target_logit = target_logit.sub(self.norm_term)
        noise_logit = noise_logit.sub(self.norm_term)
        # import pdb; pdb.set_trace()
        return target_logit, noise_logit

    def get_score(self, target_idx, noise_idx, input, embs, *args, **kwargs):
        """Get the target and noise score
        Usually logits are used as score.
        This method should be override by inherit classes
        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)
        # import pdb; pdb.set_trace()
        target_batch = embs[target_idx]
        # import pdb; pdb.set_trace()
        # target_bias = self.bias.index_select(0, target_idx)  # N
        target_score = torch.sum(input * target_batch, dim=1) # N X E * N X E

        noise_batch = embs[noise_idx]  # Nr X H
        noise_score = torch.matmul(
            input, noise_batch.t()
        )
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, *args, **kwargs):
        """Get the conventional CrossEntropyLoss
        The returned loss should be of the same size of `target`
        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class
        Returns:
            - loss: the estimated loss for each target
        """
        raise NotImplementedError()

    def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the classification loss given all four probabilities
        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution
        Returns:
            - loss: a mis-classification loss for every single case
        """

        # NOTE: prob <= 1 is not guaranteed
        logit_model = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # predicted probability of the word comes from true data distribution
        # The posterior can be computed as following
        # p_true = logit_model.exp() / (logit_model.exp() + self.noise_ratio * logit_noise.exp())
        # For numeric stability we compute the logits of true label and
        # directly use bce_with_logits.
        # Ref https://pytorch.org/docs/stable/nn.html?highlight=bce#torch.nn.BCEWithLogitsLoss
        logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

        label = torch.zeros_like(logit_model)
        label[:, :, 0] = 1

        loss = self.bce_with_logits(logit_true, label).sum(dim=2)
        return loss

    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        ori_logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # subtract Q for correction of biased sampling
        logits = ori_logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()

        if self.beta == 0:
            loss = self.ce(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).view_as(labels)

        if self.beta != 0:
            x = ori_logits.view(-1, ori_logits.size(-1))
            x = x - torch.max(x, dim = -1)[0].unsqueeze(-1)
            pos = torch.exp(x[:,0])
            neg = torch.exp(x[:,1:])
            imp = (self.beta * x[:,1:] -  torch.max(self.beta * x[:,1:],dim = -1)[0].unsqueeze(-1)).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            if torch.isinf(reweight_neg).any() or torch.isnan(reweight_neg).any():
                import pdb; pdb.set_trace()
            Ng = reweight_neg

            stable_logsoftmax = -(x[:,0] - torch.log(pos + Ng))
            loss = torch.unsqueeze(stable_logsoftmax, 1)

        return loss
    

def build_noise(number, args=None):
    if args.sample_prob == 0:
        return build_uniform_noise(number)
    if args.sample_prob == 1:
        return build_log_noise(number)

def build_log_noise(number):
    total = number
    freq = torch.Tensor([1.0] * number).cuda()
    noise = freq / total
    for i in range(number):
        noise[i] = (np.log(i + 2) - np.log(i + 1)) / np.log(number + 1)

    assert abs(noise.sum() - 1) < 0.001
    return noise

def build_uniform_noise(number):
    total = number
    freq = torch.Tensor([1.0] * number).cuda()
    noise = freq / total 
    assert abs(noise.sum() - 1) < 0.001
    return noise

import time
from torch.nn.init import xavier_uniform_, xavier_normal_

class BasicModel(nn.Module):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=50, beta=0):
        super(BasicModel, self).__init__()
        self.name = 'base'
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.item_num = item_num
        self.seq_len = seq_len
        self.beta = beta
        self.embeddings = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.interest_num = 0

    def set_device(self, device):
        self.device = device

    def set_sampler(self, args, device=None):
        self.is_sampler = True
        if args.sampled_n == 0:
            self.is_sampler = False
            return

        self.sampled_n = args.sampled_n

        noise = build_noise(self.item_num, args)

        self.sample_loss = NCELoss(noise=noise,
                                       noise_ratio=self.sampled_n,
                                       norm_term=0,
                                       reduction='elementwise_mean',
                                       per_word=False,
                                       loss_type=args.sampled_loss,
                                       beta=self.beta,
                                       device=device
                                       )

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            torch.nn.init.kaiming_normal_(weight)


    def read_out(self, user_eb, label_eb):

        # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
        atten = torch.matmul(user_eb, # shape=(batch_size, interest_num, hidden_size)
                        torch.reshape(label_eb, (-1, self.hidden_size, 1)) # shape=(batch_size, hidden_size, 1)
                        ) # shape=(batch_size, interest_num, 1)

        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1), dim=-1) # shape=(batch_size, interest_num)

        if self.hard_readout: # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                        (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0], device=user_eb.device) * self.interest_num).long()]
        else: # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), # shape=(batch_size, 1, interest_num)
                                user_eb # shape=(batch_size, interest_num, hidden_size)
                                ) # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size)) # shape=(batch_size, hidden_size)
        # readout是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
        selection = torch.argmax(atten, dim=-1)
        return readout, selection
    

    def calculate_score(self, user_eb):
        all_items = self.embeddings.weight
        scores = torch.matmul(user_eb, all_items.transpose(1, 0)) # [b, n]
        return scores


    def output_items(self):
        return self.embeddings.weight

    def calculate_full_loss(self, loss_fn, scores, target, interests):
        return loss_fn(scores, target)


    def calculate_sampled_loss(self, readout, pos_items, selection, interests):
        return self.sample_loss(pos_items.unsqueeze(-1), readout, self.embeddings.weight)


import numpy as np
import random
import math
class LogUniformSampler(object):
    def __init__(self, ntokens, device):

        self.N = ntokens
        self.prob = [0] * self.N

        self.generate_distribution()
        self.prob_tensor = torch.tensor(self.prob)
        self.cans = torch.arange(0, self.N)

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def probability(self, idx):
        return self.prob[idx]

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = dict()

        for idx in range(len(samples)):
            sample_dict[samples[idx]] = idx

        result = list()
        for idx in range(len(labels)):
            if labels[idx] in sample_dict:
                result.append((idx, sample_dict[labels[idx]]))

        return result

    def sample(self, size, labels):
        log_N = np.log(self.N)

        x = np.random.uniform(low=0.0, high=1.0, size=size)
        value = np.floor(np.exp(x * log_N)).astype(int) - 1
        samples = value.tolist()

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)
        if random.random() < 0.0002:
            print('By softmax', [round(i, 3) for i in true_freq], [round(i, 3) for i in sample_freq])

        return samples, true_freq, sample_freq

    def sample_uniform_prob(self, size, labels):
        idx = self.prob_tensor.multinomial(num_samples=size, replacement=False)
        b = self.cans[idx]

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, b)
        if random.random() < 0.0002:
            print('By uniform prob', [round(i, 3) for i in true_freq], [round(i, 3) for i in sample_freq])

        return b, true_freq, sample_freq

    def sample_uniform(self, size, labels):
        indice = random.sample(range(self.N), size)
        indice = torch.tensor(indice)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, indice)
        # print('By uniform', true_freq, sample_freq)

        return indice, true_freq, sample_freq

    def sample_unique(self, size, labels):
        # Slow. Not Recommended.
        log_N = np.log(self.N)
        samples = list()

        while (len(samples) < size):
            x = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            value = np.floor(np.exp(x * log_N)).astype(int) - 1
            if value in samples:
                continue
            else:
                samples.append(value)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq


class CapsuleNetwork(nn.Module):

    def __init__(self, hidden_size, seq_len, bilinear_type=2, interest_num=4, routing_times=3, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size # h
        self.seq_len = seq_len # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.ReLU()
            )
        if self.bilinear_type == 0: # MIND
            self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size * self.interest_num, bias=False)
        else: # ComiRec_DR
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.hidden_size, self.hidden_size))
        

    def forward(self, item_eb, mask, device):
        if self.bilinear_type == 0: # MIND
            item_eb_hat = self.linear(item_eb) # [b, s, h]
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num) # [b, s, h*in]
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else: # ComiRec_DR
            u = torch.unsqueeze(item_eb, dim=2) # shape=(batch_size, maxlen, 1, embedding_dim)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u, dim=3) # shape=(batch_size, maxlen, hidden_size*interest_num)
        
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.hidden_size))

        # [b, in, s, h]
        if self.stop_grad: # 截断反向传播，item_emb_hat不计入梯度计算中
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        # b的shape=(b, in, s)
        if self.bilinear_type > 0: # b初始化为0（一般的胶囊网络算法）
            capsule_weight = torch.zeros(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=device, requires_grad=False)
        else: # MIND使用高斯分布随机初始化b
            capsule_weight = torch.randn(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=device, requires_grad=False)

        for i in range(self.routing_times): # 动态路由传播3次
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1) # [b, in, s]
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            # 计算c，进行mask，最后shape=[b, in, 1, s]
            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight) # mask
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                # s=c*u_hat , (batch_size, interest_num, 1, seq_len) * (batch_size, interest_num, seq_len, hidden_size)
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter) # shape=(batch_size, interest_num, 1, hidden_size)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True) # shape=(batch_size, interest_num, 1, 1)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9) # shape同上
                interest_capsule = scalar_factor * interest_capsule # squash(s)->v,shape=(batch_size, interest_num, 1, hidden_size)

                # 更新b
                delta_weight = torch.matmul(item_eb_hat_iter, # shape=(batch_size, interest_num, seq_len, hidden_size)
                                        torch.transpose(interest_capsule, 2, 3).contiguous() # shape=(batch_size, interest_num, hidden_size, 1)
                                        ) # u_hat*v, shape=(batch_size, interest_num, seq_len, 1)
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len)) # shape=(batch_size, interest_num, seq_len)
                capsule_weight = capsule_weight + delta_weight # 更新b
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))

        if self.relu_layer: # MIND模型使用book数据库时，使用relu_layer
            interest_capsule = self.relu(interest_capsule)
        
        return interest_capsule
