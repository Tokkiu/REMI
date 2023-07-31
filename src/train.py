import os
import sys

pid = os.getpid()
print('pid:%d' % (pid))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import torch
from utils import get_parser, setup_seed
from evalution import train, test, output


if __name__ == '__main__':
    print(sys.argv)
    parser = get_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.gpu = '0'
    if args.gpu:
        device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
        print("use cuda:"+args.gpu if torch.cuda.is_available() else "use cpu, cuda:"+args.gpu+" not available")
    else:
        device = torch.device("cpu")
        print("use cpu")
    
    SEED = args.random_seed
    setup_seed(SEED)

    if args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367982 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 1000
    if args.dataset == 'bookv':
        path = './data/bookv_data/'
        item_count = 703121 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 1000
    # behaviors:  27158711
    if args.dataset == 'bookr':
        path = './data/bookr_data/'
        item_count = 1163015 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 1000
    # behaviors:  28723363
    if args.dataset == 'gowalla':
        path = './data/gowalla_data/'
        item_count = 174605 + 1
        batch_size = 256
        seq_len = 40
        test_iter = 1000
    if args.dataset == 'gowalla10':
        path = './data/gowalla10_data/'
        item_count = 57445 + 1
        batch_size = 256
        seq_len = 40
        test_iter = 1000
        # behaviors:  2061264
    elif args.dataset == 'familyTV':
        path = './data/familyTV_data/'
        item_count = 867632 + 1
        batch_size = 256
        seq_len = 30
        test_iter = 1000
    elif args.dataset == 'kindle':
        path = './data/kindle_data/'
        item_count = 260154 + 1
        batch_size = 128
        seq_len = 20
        test_iter = 200
    elif args.dataset == 'taobao':
        batch_size = 256
        seq_len = 50
        test_iter = 500
        path = './data/taobao_data/'
        item_count = 1708531
    elif args.dataset == 'cloth':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/cloth_data/'
        item_count = 737822 + 1

    elif args.dataset == 'tmall':
        batch_size = 256
        seq_len = 100
        test_iter = 200
        path = './data/tmall_data/'
        item_count = 946102 + 1
    elif args.dataset == 'rocket':
        batch_size = 256
        seq_len = 20
        test_iter = 200
        path = './data/rocket_data/'
        item_count = 81635 + 1

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    print("Param dataset=" + str(args.dataset))
    print("Param model_type=" + str(args.model_type))
    print("Param hidden_size=" + str(args.hidden_size))
    print("Param dropout=" + str(args.dropout))
    print("Param layers=" + str(args.layers))
    print("Param interest_num=" + str(args.interest_num))
    print("Param add_pos=" + str(args.add_pos == 1))

    print("Param weight_decay=" + str(args.weight_decay))

    batch_size = 128

    prob_dic = {
        0: 'uniform',
        1: 'log'
    }

    print("Param sampled_n=" + str(args.sampled_n))
    print("Param beta=" + str(args.rbeta))
    print("Param sampled_loss=" + str(args.sampled_loss))
    print("Param sample_prob=" + prob_dic[args.sample_prob])


    if args.p == 'train':
        train(device=device, train_file=train_file, valid_file=valid_file, test_file=test_file, 
                dataset=dataset, model_type=args.model_type, item_count=item_count, batch_size=batch_size,
                lr=args.learning_rate, seq_len=seq_len, hidden_size=args.hidden_size, 
                interest_num=args.interest_num, topN=args.topN, max_iter=args.max_iter, test_iter=test_iter, 
                decay_step=args.lr_dc_step, lr_decay=args.lr_dc, patience=args.patience, exp=args.exp, args=args)
    elif args.p == 'test':
        test(device=device, test_file=test_file, cate_file=cate_file, dataset=dataset, model_type=args.model_type, 
                item_count=item_count, batch_size=batch_size, lr=args.learning_rate, seq_len=seq_len, 
                hidden_size=args.hidden_size, interest_num=args.interest_num, topN=args.topN, coef=args.coef, exp=args.exp)
    elif args.p == 'output':
        output(device=device, dataset=dataset, model_type=args.model_type, item_count=item_count, 
                batch_size=batch_size, lr=args.learning_rate, seq_len=seq_len, hidden_size=args.hidden_size, 
                interest_num=args.interest_num, topN=args.topN, exp=args.exp)
    else:
        print('do nothing...')
    
