# -*- coding: utf-8 -*-
from model import MTDKF
from utils_ import *
import warnings
import torch_geometric
warnings.filterwarnings("ignore")

def Train(train_data, test_data, in_size, args, hg, features, att_input):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    val_data_pos = test_data[np.where(test_data[:, -1] == 1)]

    shuffle_index = np.random.choice(range(len(test_data)), len(test_data), replace=False)
    task_test_data = test_data[shuffle_index]

    model = MTDKF(
        meta_paths=args.metapaths,
        test_data=val_data_pos,
        in_size=in_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        etypes=args.etypes)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    myloss = Myloss().to(device)
    mrr = MRR()

    matrix = Matrix()
    trainloss = []
    result_list = []

    hits_max_matrix = np.zeros((1, 3))
    NDCG_max_matrix = np.zeros((1, 3))
    patience_num_matrix = np.zeros((1, 1))
    MRR_max_matrix = np.zeros((1, 1))
    epoch_max_matrix = np.zeros((1, 1))

    features_dev = {}
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            features_dev[k] = v.to(device)
        else:
            features_dev[k] = v

    # att_input: mi_seq (LongTensor), dr_graphs (list of Data), di_attr (FloatTensor)
    mi_seq = att_input.get('mi_seq', None)
    dr_graphs = att_input.get('dr_graphs', None)
    di_attr = att_input.get('di_attr', None)

    att_input_dev = {}
    att_input_dev['mi_seq'] = mi_seq.to(device)
    att_input_dev['di_attr'] = di_attr.to(device)

    # move each drug Data.x and Data.edge_index to device (if not None)
    dr_graphs_dev = []
    for g in dr_graphs:
        if hasattr(g, 'x') and g.x is not None:
            g.x = g.x.to(device)
        if hasattr(g, 'edge_index') and g.edge_index is not None:
            g.edge_index = g.edge_index.to(device)
        dr_graphs_dev.append(g)
    att_input_dev['dr_graphs'] = dr_graphs_dev

    # move heterograph edges to device
    hg = hg.to(device)
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        score_train_predict = model(hg, features_dev, att_input_dev, train_data, args.fuse_weight)
        train_label = torch.tensor(train_data[:, 3], dtype=torch.float).unsqueeze(1).to(device)
        train_loss = myloss(score_train_predict, train_label, args.loss_gamma)
        trainloss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            score_val_predict = model(hg, features_dev, att_input_dev, task_test_data, args.fuse_weight)
            predict_val = np.squeeze(score_val_predict.detach().cpu().numpy())

            hits5, ndcg5, sample_hit5, sample_ndcg5 = matrix(5, 30, predict_val, len(val_data_pos), shuffle_index)
            hits3, ndcg3, sample_hit3, sample_ndcg3 = matrix(3, 30, predict_val, len(val_data_pos), shuffle_index)
            hits1, ndcg1, sample_hit1, sample_ndcg1 = matrix(1, 30, predict_val, len(val_data_pos), shuffle_index)
            MRR_num, sample_mrr = mrr(30, predict_val, len(val_data_pos), shuffle_index)

            result = [hits5] + [hits3] + [hits1] + [ndcg5] + [ndcg3] + [ndcg1] + [MRR_num]
            result_list.append(result)
            print(f"Epoch: {epoch + 1} Train loss:{train_loss.item():.4f} "
                  f"Hits@5:{hits5:.6f} Hits@3:{hits3:.6f} Hits@1:{hits1:.6f} "
                  f"NDCG@5:{ndcg5:.6f} NDCG@3:{ndcg3:.6f} NDCG@1:{ndcg1:.6f} "
                  f"MRR:{MRR_num:.6f}")

            patience_num_matrix = ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix,
                                            epoch_max_matrix,
                                            epoch, hits1, hits3, hits5, ndcg1, ndcg3, ndcg5, MRR_num)

            if patience_num_matrix[0][0] >= args.patience:
                break
    max_epoch = int(epoch_max_matrix[0][0])
    print('Saving train result：', result_list[max_epoch][:])
    print('the optimal epoch', max_epoch)

    return result_list[max_epoch][:]
