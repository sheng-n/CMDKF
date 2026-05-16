import warnings
from train import Train
from utils_ import *
import time
from data_process import load_all
from utils_ import parameters_set

args = parameters_set()
warnings.filterwarnings("ignore")


def main_CV():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_data = load_all()
    features = all_data['features']
    in_size = all_data['in_size']

    att_input = {
        'mi_seq': all_data['mi_seq_tensor'],
        'dr_graphs': all_data['dr_graphs'],
        'di_attr': all_data['di_attr_tensor']
    }
    Hits_5, Hits_3, Hits_1, NDCG_5, NDCG_3, NDCG_1, MRR = [list() for _ in range(7)]
    fold_num = 0

    for i in range(5):
        fold_num += 1
        train_data_pos = np.array(
            pd.read_csv(args.cv_data + str(fold_num) + '/train_data_pos.csv', header=None))
        train_data_neg = np.array(
            pd.read_csv(args.cv_data + str(fold_num) + '/train_data_neg.csv', header=None))
        val_data_pos = np.array(
            pd.read_csv(args.cv_data + str(fold_num) + '/val_data_pos.csv', header=None))
        val_data_neg = np.array(
            pd.read_csv(args.cv_data + str(fold_num) + '/val_data_neg.csv',header=None))
        # print("5-cv",train_data_pos.shape, train_data_neg.shape, val_data_pos.shape, val_data_neg.shape)

        hg = construct_hg(train_data_pos)
        train_data = np.vstack((train_data_pos, train_data_neg))
        np.random.shuffle(train_data)

        val_data = np.vstack((val_data_pos, val_data_neg))
        result = Train(train_data, val_data, in_size, args, hg, features, att_input)

        Hits_5.append(result[0])
        Hits_3.append(result[1])
        Hits_1.append(result[2])
        NDCG_5.append(result[3])
        NDCG_3.append(result[4])
        NDCG_1.append(result[5])
        MRR.append(result[6])
    print('----------5 fold CV finished-----------')

    print('5-fold CV result：''Hits@5:%.6f' % np.mean(Hits_5), 'Hits@3:%.6f' % np.mean(Hits_3),
                  'Hits@1:%.6f' % np.mean(Hits_1), 'NDCG@5:%.6f' % np.mean(NDCG_5), 'NDCG@3:%.6f' % np.mean(NDCG_3),
                  'NDCG@1:%.6f' % np.mean(NDCG_1),'MRR:%.6f' % np.mean(MRR))
    return np.mean(Hits_5), np.mean(Hits_3),np.mean(Hits_1),np.mean(NDCG_5),np.mean(NDCG_3),np.mean(NDCG_1),np.mean(MRR)

def main_indep():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_data = load_all()
    features = all_data['features']
    in_size = all_data['in_size']
    att_input = {
        'mi_seq': all_data['mi_seq_tensor'],
        'dr_graphs': all_data['dr_graphs'],
        'di_attr': all_data['di_attr_tensor']
    }

    Hits_5, Hits_3, Hits_1, NDCG_5, NDCG_3, NDCG_1, MRR = [list() for _ in range(7)] #

    train_data_pos = np.array(
        pd.read_csv(args.indepent_data + '/train_data_pos.csv', header=None))
    train_data_neg = np.array(
        pd.read_csv(args.indepent_data + '/train_data_neg.csv',header=None))
    val_data_pos = np.array(
        pd.read_csv(args.indepent_data + '/test_data_pos.csv', header=None))
    val_data_neg = np.array(
        pd.read_csv(args.indepent_data + '/test_data_neg.csv', header=None))

    hg = construct_hg(train_data_pos)
    print("data loaded", train_data_pos.shape, train_data_neg.shape, val_data_pos.shape, val_data_neg.shape)
    train_data = np.vstack((train_data_pos, train_data_neg))
    np.random.shuffle(train_data)

    val_data = np.vstack((val_data_pos, val_data_neg))
    result = Train(train_data, val_data, in_size, args, hg, features, att_input)

    Hits_5.append(result[0])
    Hits_3.append(result[1])
    Hits_1.append(result[2])
    NDCG_5.append(result[3])
    NDCG_3.append(result[4])
    NDCG_1.append(result[5])
    MRR.append(result[6])
    print('----------independent test finished-----------')
    print('Independent test result：''Hits@5:%.6f' % np.mean(Hits_5), 'Hits@3:%.6f' % np.mean(Hits_3),
                  'Hits@1:%.6f' % np.mean(Hits_1), 'NDCG@5:%.6f' % np.mean(NDCG_5), 'NDCG@3:%.6f' % np.mean(NDCG_3),
                  'NDCG@1:%.6f' % np.mean(NDCG_1),'MRR:%.6f' % np.mean(MRR))
    return np.mean(Hits_5), np.mean(Hits_3),np.mean(Hits_1),np.mean(NDCG_5),np.mean(NDCG_3),np.mean(NDCG_1),np.mean(MRR)



if __name__ == '__main__':
    total_start_time = time.time()

    args = parameters_set()
    print('Starting the 5-fold CV experiment')

    cv_start_time = time.time()
    CV_Hits5, CV_Hits3, CV_Hits1, CV_NDCG_5, CV_NDCG_3, CV_NDCG_1, CV_MRR_num = main_CV()
    cv_end_time = time.time()
    cv_time = cv_end_time - cv_start_time

    print('Starting the independent test experiment')
    indep_start_time = time.time()
    indep_Hits5, indep_Hits3, indep_Hits1, indep_NDCG_5, indep_NDCG_3, indep_NDCG_1, indep_MRR_num = main_indep()
    indep_end_time = time.time()
    indep_time = indep_end_time - indep_start_time

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    print("\n" + "=" * 50)
    print("=" * 50)
    print(f"5-cv times: {cv_time:.2f} sec ({cv_time / 60:.2f} )")
    print(f"total times: {total_time:.2f} sec ({total_time / 60:.2f} min)")






