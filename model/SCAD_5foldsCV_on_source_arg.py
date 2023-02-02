import argparse
parser = argparse.ArgumentParser(description='feature extractor and drug name')
parser.add_argument('--feature_extractor','-e',help='feature extractor')
parser.add_argument('--drug_name','-d',help='drug name')
parser.add_argument('--geneset','-g',help='geneset, should be either one of <_norm, _norm_ppi, _norm_tp4k>')
parser.add_argument('--adjust','-adj',help='should be either one of <Y, N>; use randomly selected 5k single cells to help adjust the feature distribution between source and target domain, default is not adjust',nargs='?',default='N')
parser.add_argument('--cell_lines','-l',help='use all cell lines or solid cell line only, default False', nargs='?', default="all")

args = parser.parse_args()
extractor = args.feature_extractor
DRUG = args.drug_name
geneset = args.geneset
adjust = args.adjust
cell_lines = args.cell_lines

import torch
import numpy as np
import pandas as pd
from numpy import random
from sklearn.metrics import *
import sklearn.preprocessing as sk

init_seed = 42
import os
import sys
random.seed(init_seed)
np.random.seed(init_seed)
os.environ['PYTHONHASHSEED'] = str(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
import random
from sklearn.model_selection import StratifiedKFold
import itertools
from itertools import cycle
from SCADmodules import *

###############################
#    set working directory    #
###############################
WDR = 'D:/Project2'
os.chdir(WDR+"/ANN/BMDSTL/BM")
sys.path.append(WDR+"/ANN/BMDSTL/BM")

feature=geneset
if adjust=="Y":
    X_5k_cells_raw = pd.read_csv("./data/Pancan_CCLE_10X/scrna_ccle_CPM_random_5k_raw_x_scaled_ncbi.tsv",sep='\t', index_col=0, decimal='.')
    feature=geneset+'_adj'

tr_eva_on_source_summary_path = './results'+feature+'/train_eval_on_source_hypers_summary_5fds_fixlr.txt'

if cell_lines == "solid":
    tr_eva_on_source_summary_path = './results_solid'+feature+'/train_eval_on_source_hypers_summary_5fds_fixlr.txt'

##use GPU##
GPU = True

if GPU:
    device = "cuda"
else:
    device = "cpu"

#######################################################
#                 DRUG, SAVE, LOAD                    #
#######################################################
MAX_ITER = 20
MODE = "fine_tune/" + extractor

if cell_lines!="solid":
    SAVE_RESULTS_TO = "./results"+feature+"/" + DRUG + "/" + MODE + "/"
    SAVE_TRACE_TO = "./results"+feature+"/" + DRUG + "/" + MODE + "/trace/"
    SOURCE_DIR = 'source_5_folds'

TARGET_DIR = 'target_5_folds'

if cell_lines=="solid":
    SAVE_RESULTS_TO = "./results_solid"+feature+"/" + DRUG + "/" + MODE + "/"
    SAVE_TRACE_TO = "./results_solid"+feature+"/" + DRUG + "/" + MODE + "/trace/"
    SOURCE_DIR = 'source_solid_5_folds'

LOAD_DATA_FROM = 'data/split'+geneset+'/' + DRUG + '/stratified/'    #####Load data from

dirName = SAVE_RESULTS_TO + 'model/'

if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")

dirName = SAVE_RESULTS_TO + 'test/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")

#######################################################
#                    FUNCTIONS                        #
#######################################################
def predict_label(XTestCells, gen_model, map_model):
    """
    Inputs:
    :param XTestCells - X_target_test
    :param gen_model    - current FX model
    :param map_model    - current MTL model

    Output:
        - Predicted (binary) labels (Y for input target test data)
    """

    gen_model.eval()
    gen_model.to(device)
    map_model.eval()
    map_model.to(device)

    if extractor == 'FX' or extractor == 'FX_MLP':
        F_xt_test = gen_model(XTestCells.to(device))

    else:
        raise NotImplementedError('{} is not a valid extractor'.format(extractor))

    yhatt_test = map_model(F_xt_test)
    return yhatt_test

def evaluate_model(XTestCells, YTestCells, gen_model, map_model):
    """
    Inputs:
    :param XTestCells - Single cell test data
    :param YTestCells - true class labels (binary) for Single cell test data
    :param path_to_models - path to the saved models from training

    Outputs:
        - test loss
        - test accuracy (AUC)
    """
    XTestCells = XTestCells.to(device)
    YTestCells = YTestCells.to(device)
    y_predicted = predict_label(XTestCells, gen_model, map_model)

    # #LOSSES
    C_loss_eval = torch.nn.BCELoss()
    closs_test = C_loss_eval(y_predicted, YTestCells)

    if device == "cuda":
        YTestCells = YTestCells.to("cpu")
    yt_true_test = YTestCells.view(-1, 1)
    yt_true_test = yt_true_test.cpu()
    y_predicted = y_predicted.cpu()

    AUC_test = roc_auc_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())
    return closs_test, AUC_test

def roc_auc_score_trainval(y_true, y_predicted):
    # To handle the case where we only have training samples of one class
    # in our mini-batch when training since roc_auc_score 'breaks' when
    # there is only one class present in y_true:
    # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.

    # The following code is taken from
    # https://stackoverflow.com/questions/45139163/roc-auc-score-only-one-class-present-in-y-true?rq=1 #
    if len(np.unique(y_true)) == 1:
        return accuracy_score(y_true, np.rint(y_predicted))
    return roc_auc_score(y_true, y_predicted)

#######################################################
#                Hyper-Parameter Lists                #
#######################################################
ls_splits = ['split1', 'split2', 'split3','split4','split5']
ls_mb_size = [{'mbS': 8, 'mbT': 8},\
        {'mbS': 32, 'mbT': 32}]

ls_h_dim = [1024, 512]
ls_z_dim = [256, 128]

ls_lr = [0.001]
ls_epoch = [10, 20, 30, 40, 60, 80, 100]
ls_lam = [0.2, 0.6, 0.8, 1.0, 2, 3, 4, 5]

ls_dropout_gen = [0.5]
ls_dropout_mtl = ls_dropout_gen
ls_dropout_dg = ls_dropout_gen

#############
import pandas as pd
YGDSC = pd.read_csv("./data/split_norm/" + DRUG + '/' + 'Source_exprs_resp_z.'+DRUG+'.tsv',
                                    sep='\t', index_col=0, decimal='.')

if cell_lines=="solid":
    YGDSC = pd.read_csv("./data/split_norm/" + DRUG + '/' + 'Source_solid_exprs_resp_z.'+DRUG+'.tsv',
            sep='\t', index_col=0, decimal='.')

from collections import Counter
class_sample_count = np.array([Counter(YGDSC['response'])[0]/len(YGDSC['response']),Counter(YGDSC['response'])[1]/len(YGDSC['response'])])

#######################################################
#         AITL Model Training Starts Here             #
#######################################################
for index, mbsize in enumerate(ls_mb_size):
    mbS = mbsize['mbS']
    mbT = mbsize['mbT']
    for iters in range(MAX_ITER):
        print("\n\n\nITERATION # {}".format(iters))
        print("-----------------------------\n\n")

        # Randomly selecting hyper-parameter values #
        hdm = random.choice(ls_h_dim)
        zdm = random.choice(ls_z_dim)
        lrs = random.choice(ls_lr)
        epch = random.choice(ls_epoch)
        lambd1 = random.choice(ls_lam)

        drop_gen = random.choice(ls_dropout_gen)
        drop_mtl = random.choice(ls_dropout_mtl)
        drop_dg = random.choice(ls_dropout_dg)

        h_dim = hdm
        z_dim = zdm
        lr = lrs
        epoch = epch
        lam1 = lambd1
        dropout_gen = drop_gen
        dropout_mtl = drop_gen
        dropout_dg = drop_gen

        print("-- Parameters used: --")
        print("h_dim: {}\nz_dim: {}\nlr: {}\nepoch: {}\nlambda1: {}".format(
                h_dim,
                z_dim,
                lr,
                epoch,
                lam1))

        print(
            "mbS: {}\nmbT: {}\ndropout_gen: {}\ndropout_mtl: {}\ndropout_dg: {}\n".format(
                mbS,
                mbT,
                dropout_gen,
                dropout_mtl,
                dropout_dg))

        batch_sizes = 'mbS' + str(mbS) + '_mbT' + str(mbT)
        hyper_para_record = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(
            epoch) + '_lambda1' + str(lam1)  + '_dropouts' + str(dropout_gen) \
                            + '_' + str(dropout_mtl) + '_' + str(dropout_dg)  + '_mbS' + str(mbS) + '_mbT' + str(mbT)

        ###record valiation result
        AUCval_source_splits_total = []  ## on source set

        for split in ls_splits:  # for each split
            AUCval_source_1split = []

            print("\n\nReading data for {} ...\n".format(split))
            # Loading Source Data #
            XTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_train_source.tsv',
                                            sep='\t', index_col=0, decimal='.')
            YTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_train_source.tsv',
                                            sep='\t', index_col=0, decimal='.')
            XValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_val_source.tsv',
                                            sep='\t', index_col=0, decimal='.')
            YValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_val_source.tsv',
                                            sep='\t', index_col=0, decimal='.')

            # Loading Target single cell Data #
            XTrainCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/X_train_target.tsv',
                                            sep='\t', index_col=0, decimal='.')
            YTrainCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/Y_train_target.tsv',
                                            sep='\t', index_col=0, decimal='.')

            XTestCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/X_test_target.tsv',
                                            sep='\t', index_col=0, decimal='.')
            YTestCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/Y_test_target.tsv',
                                            sep='\t', index_col=0, decimal='.')


            # Load randomly sampled Data #
            if adjust == "Y":
                X_5k_cells = X_5k_cells_raw.transpose()
                X_5k_cells.columns = X_5k_cells.columns.astype(str)
                intersect_gene = [value for value in XValGDSC.columns if value in X_5k_cells.columns]
                X_5k = X_5k_cells[intersect_gene]
                XTrainGDSC = XTrainGDSC[intersect_gene]
                XValGDSC = XValGDSC[intersect_gene]
                XTrainCells = XTrainCells[intersect_gene]
                XTestCells = XTestCells[intersect_gene]


            print("Data successfully read!")

            model_params = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(
                epoch) + '_lamb1' + str(lam1) \
                           + '_dropouts' + str(dropout_gen) + '_' + str(dropout_mtl) + '_' + str(
                 dropout_dg) + '_mbS' + str(mbS) + '_mbT' + str(mbT)

            dirName = SAVE_TRACE_TO + batch_sizes + '/' + model_params + '/'
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Directory ", dirName, " Created ")
            else:
                print("Directory ", dirName, " already exists")

                print("\n\n-- Reading data of {} ... --".format(split))

            # Temporarily combine Source training data and Target training data
            # to fit standard scaler on gene expression of combined training data.
            # Then, apply fitted scaler to (and transform) Source validation,
            # Target validation, and Target test (e.g. normalize validation and test
            # data of source and target with respect to source and target train)r
            XTrainCombined = pd.concat([XTrainGDSC, XTrainCells])

            if adjust == "Y":
                XTrainCombined = pd.concat([XTrainCombined, X_5k_cells[intersect_gene]])

            scalerTrain = sk.StandardScaler()
            scalerTrain.fit(XTrainCombined.values)
            XTrainGDSC_N = scalerTrain.transform(XTrainGDSC.values)
            XValGDSC_N = scalerTrain.transform(XValGDSC.values)

            XTrainCells_N = scalerTrain.transform(XTrainCells.values)
            XTestCells_N = scalerTrain.transform(XTestCells.values)
            ##
            TXValGDSC_N = torch.FloatTensor(XValGDSC_N)
            TXValGDSC_N = TXValGDSC_N.to(device)

            TYValGDSC = torch.FloatTensor(YValGDSC.values)
            TYValGDSC = TYValGDSC.to(device)

            TXTestCells_N = torch.FloatTensor(XTestCells_N)
            TYTestCells = torch.FloatTensor(YTestCells.values.astype(int))
            TXTestCells_N = TXTestCells_N.to(device)
            TYTestCells = TYTestCells.to(device)

            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in YTrainGDSC.values])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.reshape(-1)  # Flatten out the weights so it's a 1-D tensor of weights
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)

            # Apply sampler on XTrainCells_N
            # GDSC cell line dataset ##
            CDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainGDSC_N), torch.FloatTensor(YTrainGDSC.values))

            PDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainCells_N), torch.FloatTensor(YTrainCells.values.astype(int)))
            # upsampling source domain training set which is unbalanced##
            CLoader = torch.utils.data.DataLoader(dataset=CDataset, batch_size=mbS, shuffle=False, sampler=sampler)
            PLoader = torch.utils.data.DataLoader(dataset=PDataset, batch_size=mbT, shuffle=True)

            n_sample, IE_dim = XTrainGDSC_N.shape

            if extractor == 'FX' or extractor == 'FX_MLP':
                Gen = FX(dropout_gen, IE_dim, h_dim, z_dim)
            else:
                raise NotImplementedError('{} is not a valid extractor'.format(extractor))

            Map = MTLP(dropout_mtl, h_dim, z_dim)
            DG = Discriminator(dropout_dg, h_dim, z_dim)
            Gen.to(device)
            Map.to(device)
            DG.to(device)

            optimizer_2 = torch.optim.Adagrad(
                itertools.chain(Gen.parameters(), Map.parameters(), DG.parameters()),lr=lr)

            C_loss = torch.nn.BCELoss()

            l1 = []
            l2 = []
            classif = []
            L = []  # total loss
            DG_losstr = []
            DG_auctr = []

            for it in range(epoch):
                epoch_cost1ls = []
                epoch_cost2ls = []
                epoch_classifls = []
                epoch_DGloss = []
                epoch_DGauc = []

                AUCvals_source = [] #####validation AUC on source set
                classif_lossval = []

                epoch_loss = []
                for i, data in enumerate(zip(cycle(CLoader), PLoader)):
                    DataS = data[0]
                    DataT = data[1]
                    ## Sending data to device = cuda/cpu
                    xs = DataS[0].to(device)  ## source data x
                    ys = DataS[1].view(-1, 1).to(device)  ## source data y
                    xt = DataT[0].to(device)  ## target data x
                    yt = DataT[1].view(-1, 1).to(device)  ## target data y

                    # Skip to next set of training batch if any of xs or xt has less
                    # than a certain threshold of training examples. Let such threshold
                    # be 5 for now
                    if xs.size()[0] < 5 or xt.size()[0] < 5:
                        continue

                    Gen.train()
                    Map.train()
                    DG.train()

                    if extractor == 'FX' or extractor == 'FX_MLP':
                        F_xs = Gen(xs)
                        F_xt = Gen(xt)

                    else:
                        raise NotImplementedError('{} is not a valid extractor'.format(extractor))

                    yhat_xs = Map(F_xs)

                    closs = C_loss(yhat_xs, ys)
                    loss1 = closs

                    Labels = torch.ones(F_xs.size(0), 1)
                    Labelt = torch.zeros(F_xt.size(0), 1)
                    Lst = torch.cat([Labels, Labelt], 0).to(device)
                    Xst = torch.cat([F_xs, F_xt], 0).to(device)

                    yhat_DG = DG(Xst)
                    DG_loss = C_loss(yhat_DG, Lst)

                    loss2 = lam1 * DG_loss

                    ### Loss on source training data
                    Loss = loss1 + loss2

                    optimizer_2.zero_grad()
                    Loss.backward()
                    optimizer_2.step()

                    epoch_cost1ls.append(loss1)
                    epoch_cost2ls.append(loss2)
                    epoch_classifls.append(closs)
                    epoch_loss.append(Loss)
                    epoch_DGloss.append(DG_loss)

                    ## y_true of target domain ##
                    y_true = yt.view(-1, 1)
                    y_true = y_true.cpu()

                    y_trueDG = Lst.view(-1, 1)
                    y_predDG = yhat_DG

                    y_trueDG = y_trueDG.cpu()
                    y_predDG = y_predDG.cpu()

                    AUCDG = roc_auc_score_trainval(y_trueDG.detach().numpy(), y_predDG.detach().numpy())
                    epoch_DGauc.append(AUCDG)

                l1.append(torch.mean(torch.Tensor(epoch_cost1ls)))
                l2.append(torch.mean(torch.Tensor(epoch_cost2ls)))
                classif.append(torch.mean(torch.Tensor(epoch_classifls)))
                L.append(torch.mean(torch.FloatTensor(epoch_loss)))
                DG_losstr.append(torch.mean(torch.Tensor(epoch_DGloss)))
                DG_auctr.append(torch.mean(torch.Tensor(epoch_DGauc)))

                with torch.no_grad():

                    Gen.eval()
                    Gen.to(device)
                    Map.eval()
                    Map.to(device)
                    DG.eval()
                    DG.to(device)

                    TXValGDSC_N = TXValGDSC_N.to(device)

                    if extractor == 'FX' or extractor == 'FX_MLP':
                        F_xs_val = Gen(TXValGDSC_N)

                    else:
                        raise NotImplementedError('{} is not a valid extractor'.format(extractor))

                    yhats_val = Map(F_xs_val)  ## Map is the drug response predictor

                    ## calculate AUC for source validation data
                    ys_true_val = TYValGDSC.view(-1, 1)
                    ys_true_val = ys_true_val.cpu()
                    yhats_val = yhats_val.cpu()
                    AUC_source_val = roc_auc_score_trainval(ys_true_val.detach().numpy(), yhats_val.detach().numpy())
                    AUCvals_source.append(AUC_source_val)

                    #########record all source validation results
                    AUCval_source_splits_total.append(AUC_source_val)
                    AUCval_source_1split.append(AUC_source_val)

                # Take average across all training batches
                loss1tr_mean = l1[it]
                loss2tr_mean = l2[it]
                totlosstr_mean = L[it]

                DGlosstr_mean = DG_losstr[it]
                DGauctr_mean = DG_auctr[it]
                closstr_mean = classif[it]
                ######
                AUCval_source_mean = np.mean(np.array(AUCvals_source))
                ######

                print("\n\nEpoch: {}".format(it))
                print("(source, tr) loss1 mean: {}".format(loss1tr_mean))
                print("(source, tr) loss2 mean: {}".format(loss2tr_mean))
                print("(source, tr) total loss mean: {}".format(totlosstr_mean))
                print("(source, tr) DG loss mean: {}".format(DGlosstr_mean))
                print("(val) source AUC mean: {}".format(AUCval_source_mean))

            ##record AUC of specific split validation source set
            AUCval_source_1split = np.array(AUCval_source_1split)
            avgAUCval_source_1split = np.mean(AUCval_source_1split)
            with open(tr_eva_on_source_summary_path, 'a') as f:
                f.write("----------Drug = {}, extractor = {}, source average validation AUC of {} = {}, {}----------\n".format(DRUG, extractor, split, avgAUCval_source_1split, hyper_para_record))

        AUCval_source_splits_total = np.array(AUCval_source_splits_total)
        avgAUCval_source = np.mean(AUCval_source_splits_total)
        with open(tr_eva_on_source_summary_path, 'a') as f:
            f.write("----------Drug = {}, extractor = {}, source average validation AUC of iter{} = {}, {}----------\n".format(DRUG, extractor, iters, avgAUCval_source, hyper_para_record))

