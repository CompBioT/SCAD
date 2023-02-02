import argparse
parser = argparse.ArgumentParser(description='feature extractor and drug name')
parser.add_argument('--feature_extractor','-e',help='feature extractor')
parser.add_argument('--drug_name','-d',help='drug name')
parser.add_argument('--geneset','-g',help='geneset')
parser.add_argument('--init_seed','-s',help='init_seed',type=int)
#
parser.add_argument('--h_dim','-h_dim',help='h_dimension',type=int)
parser.add_argument('--z_dim','-z_dim',help='z_dimension',type=int)
parser.add_argument('--epoch','-ep',help='epoch',type=int)
parser.add_argument('--lam1','-la1',help='lam1',type=float)
parser.add_argument('--mbS','-mbS',help='mbS',type=int)
parser.add_argument('--mbT','-mbT',help='mbT',type=int)

#
args = parser.parse_args()
extractor = args.feature_extractor
DRUG = args.drug_name
geneset = args.geneset
init_seed = args.init_seed

h_dim = args.h_dim
z_dim = args.z_dim
epoch = args.epoch
lam1 = args.lam1
mbS = args.mbS
mbT = args.mbT

import torch
import numpy as np
from numpy import random
from sklearn.metrics import *
import pandas as pd
import sklearn.preprocessing as sk
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

from torch.utils.data.sampler import WeightedRandomSampler
import itertools
from itertools import cycle
from sklearn.metrics import average_precision_score
from captum.attr import IntegratedGradients
###############################
#    set working directory    #
###############################
os.chdir("D:/Project2/ANN/BMDSTL/BM")
sys.path.append("D:/Project2/ANN/BMDSTL/BM")
WDR = 'D:/Project2'
from SCADmodules import *
tr_ts_summary_path = "./results"+geneset+"_L/train_test_summary_5folds_5seeds.txt"
##use GPU##
GPU = True

if GPU:
    device = "cuda"
else:
    device = "cpu"

#######################################################
#                 DRUG, SAVE, LOAD                    #
#######################################################
max_iter = 1
MODE = "bimodal/best_models/" + extractor

SAVE_RESULTS_TO = "./results"+geneset+"_L/" + DRUG + "/" + MODE + "/" + 'seed' + str(init_seed) + "/"
SAVE_TRACE_TO = "./results"+geneset+"_L/" + DRUG + "/" + MODE + "/trace/" + 'seed' +  str(init_seed) + "/"
TARGET_DIR = 'target_5_folds'
SOURCE_DIR = 'source_5_folds'
LOAD_DATA_FROM = 'data/split'+geneset+'/' + DRUG + '/stratified/'    ###Load source data
LOAD_TARGET_DATA_FROM = 'data_L/split'+geneset+'/' + DRUG + '/stratified/'  ###Load target data

dirName = SAVE_RESULTS_TO + 'test/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")


#######################################################
#                      Functions                      #
#######################################################
@torch.no_grad()
def predict_label(XTestCells, gen_model, map_model):
    """
    Inputs:
    :param XTestCells - X_target_test
    :param gen_model
    :param map_model

    Output:
        - Predicted (binary) labels (Y for input target test data)
    """

    gen_model.eval()
    gen_model.to(device)
    map_model.eval()  ## predictor
    map_model.to(device)

    if extractor == 'FX' or extractor == 'FX_MLP':
        F_xt_test = gen_model(XTestCells.to(device))

    else:
        raise NotImplementedError('{} is not a valid extractor'.format(extractor))

    yhatt_test = map_model(F_xt_test)
    return yhatt_test

#https://vimsky.com/zh-tw/examples/detail/python-method-sklearn.metrics.average_precision_score.html

def evaluate_model(XTestCells, YTestCells, gen_model, map_model):
    """
    Inputs:
    :param XTestCells - single cell test data
    :param YTestCells - true class labels (binary) for single cell test data
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

    yt_true_test = YTestCells.view(-1,1)
    yt_true_test = yt_true_test.cpu()
    y_predicted = y_predicted.cpu()

    print("{} and {}".format(yt_true_test, y_predicted))   # yt_true_test = binary, y_predicted = float/decimal
    AUC_test = roc_auc_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())

    # Precision Recall
    APR_test = average_precision_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())

    return closs_test, AUC_test, APR_test, y_predicted

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

def ENTERZid2HGNCid(ncbilist):
    # self.dict_ens = {}
    dict_sym = {}
    HGNC = pd.read_csv("./data/original/HGNC_symbol_all_genes.tsv", sep='\t', index_col=0, decimal='.')
    for index in range(len(HGNC['Approved symbol'])):
        if (not isinstance(HGNC['Approved symbol'][index], float)):
            if HGNC['NCBI Gene ID'][index] == HGNC['NCBI Gene ID'][index]:
                dict_sym[int(HGNC['NCBI Gene ID'][index])] = HGNC['Approved symbol'][index]
    #
    hgnclist = []
    indexkeep = []

    for index in range(len(ncbilist)):
        if int(ncbilist[index]) in dict_sym:
            indexkeep.append(index)
            hgnclist.append(dict_sym[int(ncbilist[index])])

    return (hgnclist, indexkeep)

#######################################################
#                Hyper-Parameter Lists                #
#######################################################
ls_splits = ['split1', 'split2', 'split3','split4','split5']

ls_mb_size = [{'mbS': mbS, 'mbT': mbS}]      # Set the batch size for the best model manually

YGDSC = pd.read_csv("./data_L/split_norm/" + DRUG + '/' + 'Source_exprs_resp_z.'+DRUG+'.tsv',
                                    sep='\t', index_col=0, decimal='.')

from collections import Counter
Counter(YGDSC['response'])[0]/len(YGDSC['response'])
Counter(YGDSC['response'])[1]/len(YGDSC['response'])
class_sample_count = np.array([Counter(YGDSC['response'])[0]/len(YGDSC['response']),Counter(YGDSC['response'])[1]/len(YGDSC['response'])])
print("class_sample_count = {}".format(class_sample_count))
#######################################################
#         AITL Model Training Starts Here             #
#######################################################
for index, mbsize in enumerate(ls_mb_size):
    mbS = mbsize['mbS']
    mbT = mbsize['mbT']

    # Hard-code the values for testing and training 'best model' ##, parsed
    lr = 1e-3
    dropout_gen = 0.5
    dropout_mtl = dropout_gen
    dropout_dg = dropout_gen

    print("-- Parameters used: --")
    print("h_dim: {}\nz_dim: {}\nlr: {}\nepoch: {}\nlambda1: {}\n".format(h_dim,z_dim,lr,epoch,lam1))

    print("mbS: {}\nmbT: {}\ndropout_gen: {}\ndropout_mtl: {}\ndropout_dg: {}\n".format(mbS,
                                                                                        mbT,
                                                                                        dropout_gen,
                                                                                        dropout_mtl,
                                                                                        dropout_dg))

    batch_sizes = 'mbS' + str(mbS) + '_mbT' + str(mbT)
    test_results_name = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(epoch) + '_lambda1' + str(lam1) + \
                        '_mbS' + str(mbS) + '_mbT' + str(mbT) + '_seed' + str(init_seed) +'.tsv'

    test_predicted_y_name = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(epoch) + '_lambda1' + str(lam1) + \
                        '_mbS' + str(mbS) + '_mbT' + str(mbT) + '_seed' + str(init_seed) + '_predicted_y.tsv'

    IG_source_val_results_name = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(
        epoch) + '_lambda1' + str(lam1) + \
                        '_mbS' + str(mbS) + '_mbT' + str(mbT) + '_seed' + str(init_seed) + '_attribution_source_val.tsv'

    IG_target_test_results_name = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(
        epoch) + '_lambda1' + str(lam1) + \
                                 '_mbS' + str(mbS) + '_mbT' + str(mbT) + '_seed' + str(init_seed) + '_attribution_target_test.tsv'


    test_results_dir = SAVE_RESULTS_TO + 'test/' + batch_sizes + '/'
    dirName = test_results_dir
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    test_results_file = os.path.join(test_results_dir, test_results_name)
    if os.path.isfile(test_results_file):
        os.remove(test_results_file)

    test_predicted_y_file = os.path.join(test_results_dir, test_predicted_y_name)
    if os.path.isfile(test_predicted_y_file):
        os.remove(test_predicted_y_file)


    IG_source_val_file = os.path.join(test_results_dir, IG_source_val_results_name)
    # if os.path.isfile(IG_source_val_file):
    #     os.remove(IG_source_val_file)

    IG_target_test_file = os.path.join(test_results_dir, IG_target_test_results_name)
    if os.path.isfile(IG_target_test_file):
        os.remove(IG_target_test_file)
    #
    with open(test_results_file, 'a') as f:
        f.write("-- Parameters --\n\n")
        f.write("h_dim: {}\nz_dim: {}\nlr: {}\nepoch: {}\nlambda1: {}\nmbS: {}\nmbT: {}\n".format(h_dim,z_dim,lr,epoch,lam1,mbS,mbT))

    AUCtest_splits_total = []
    APRtest_splits_total = []

    for split in ls_splits:     # for each split

        print("\n\nReading data for {} ...\n".format(split))
        # Loading Source Data, in this model, both source and target label are binarized, the target domain labels are masked during training#
        XTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_train_source.tsv',
                                    sep='\t', index_col=0, decimal='.')
        YTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_train_source.tsv',
                                    sep='\t', index_col=0, decimal='.')
        XValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_val_source.tsv',
                                    sep='\t', index_col=0, decimal='.')
        YValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_val_source.tsv',
                                    sep='\t', index_col=0, decimal='.')

        # Loading Target (Single cell) Data, YTrainCells are not used during training#
        XTrainCells = pd.read_csv(LOAD_TARGET_DATA_FROM + TARGET_DIR + '/' + split + '/X_train_target.tsv',
                                    sep='\t', index_col=0, decimal='.')
        YTrainCells = pd.read_csv(LOAD_TARGET_DATA_FROM + TARGET_DIR + '/' + split + '/Y_train_target.tsv',
                                    sep='\t', index_col=0, decimal='.')
        XTestCells = pd.read_csv(LOAD_TARGET_DATA_FROM + TARGET_DIR + '/' + split + '/X_test_target.tsv',
                                        sep='\t', index_col=0, decimal='.')
        YTestCells = pd.read_csv(LOAD_TARGET_DATA_FROM + TARGET_DIR + '/' + split + '/Y_test_target.tsv',
                                        sep='\t', index_col=0, decimal='.')
        print("Data successfully read!")

    ####
        model_params = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(epoch) + '_lamb1' + str(lam1) \
                        + '_mbS' + str(mbS) + '_mbT' + str(mbT)

    ###
        # Temporarily combine Source training data and Target training data
        # to fit standard scaler on gene expression of combined training data.
        # Then, apply fitted scaler to (and transform) Source validation,
        # and Target test (e.g. normalize validation and test
        # data of source and target with respect to source and target train)
        XTrainCombined = pd.concat([XTrainGDSC, XTrainCells])
        scalerTrain = sk.StandardScaler()
        scalerTrain.fit(XTrainCombined.values)
        # N means Standardized
        XTrainGDSC_N = scalerTrain.transform(XTrainGDSC.values)
        XTrainCells_N = scalerTrain.transform(XTrainCells.values)
        XTestCells_N = scalerTrain.transform(XTestCells.values)

        TXTestCells_N = torch.FloatTensor(XTestCells_N)
        TYTestCells = torch.FloatTensor(YTestCells.values.astype(int))
        TXTestCells_N = TXTestCells_N.to(device)
        TYTestCells = TYTestCells.to(device)

        weight = 1. / class_sample_count
        #upsampling source domain training set which is unbalanced##
        samples_weight = np.array([weight[t] for t in YTrainGDSC.values])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.reshape(-1) # Flatten out the weights so it's a 1-D tensor of weights
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

        # Apply sampler on XTrainGDSC_N
        ## GDSC cell line dataset ##
        CDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainGDSC_N), torch.FloatTensor(YTrainGDSC.values))

        PDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainCells_N), torch.FloatTensor(YTrainCells.values.astype(int)))
        #upsampling source domain training set which is unbalanced##
        CLoader = torch.utils.data.DataLoader(dataset = CDataset, batch_size= mbS, shuffle=False, sampler = sampler)
        PLoader = torch.utils.data.DataLoader(dataset=PDataset, batch_size=mbT, shuffle=True)

        ##
        n_sample, IE_dim = XTrainGDSC_N.shape

        if extractor == 'FX':
            Gen = FX(dropout_gen, IE_dim, h_dim, z_dim)
        else:
            raise NotImplementedError('{} is not a valid extractor'.format(extractor))

        Map = MTLP(dropout_mtl, h_dim, z_dim)               #### Mapper
        DG = Discriminator(dropout_dg, h_dim, z_dim)        #### Global Discriminator####

        Gen.to(device)
        Map.to(device)
        DG.to(device)

        # feature extractor, domain adaptation and global discriminator #
        optimizer_2 = torch.optim.Adagrad(itertools.chain(Gen.parameters(), Map.parameters(), DG.parameters()),lr = lr)

        C_loss = torch.nn.BCELoss()

        l1 = []
        l2 = []
        classif = []
        L = [] # total loss
        DG_losstr = []
        DG_auctr = []

        for it in range(epoch):
            epoch_cost1ls = []
            epoch_cost2ls = []
            epoch_classifls = []
            epoch_DGloss = []
            epoch_DGauc = []

            epoch_loss = []

            for i, data in enumerate(zip(cycle(CLoader), PLoader)):
                DataS = data[0]
                DataT = data[1]
                #
                xs = DataS[0].to(device)
                ys = DataS[1].view(-1,1).to(device)
                xt = DataT[0].to(device)
                yt = DataT[1].view(-1,1).to(device)

                # Skip to next set of training batch if any of xs or xt has less
                # than a certain threshold of training examples.
                # Let such threshold be 5
                if xs.size()[0] < 5 or xt.size()[0] < 5:
                    continue

                Gen.train()          ### Gen = EN(dropout_gen, IE_dim, h_dim, z_dim)     #### MLP or other gene feature extractor
                Map.train()          ### mapper
                DG.train()           ### discriminator

                if extractor == 'FX' or extractor == 'FX_MLP':
                    F_xs = Gen(xs)
                    F_xt = Gen(xt)

                else:
                    raise NotImplementedError('{} is not a valid extractor'.format(extractor))

                yhat_xs = Map(F_xs)
                #classification for GDSC
                closs = C_loss(yhat_xs, ys)
                loss1 = closs

                Labels = torch.ones(F_xs.size(0), 1)
                Labelt = torch.zeros(F_xt.size(0), 1)
                Lst = torch.cat([Labels, Labelt],0).to(device)  ## combine domain labels
                Xst = torch.cat([F_xs, F_xt], 0).to(device)     ##combine domain data

                yhat_DG = DG(Xst)       ##predicted domain label from global discriminator
                DG_loss = C_loss(yhat_DG, Lst)

                loss2 = lam1 * DG_loss

                Loss = loss1 + loss2
                ###

                optimizer_2.zero_grad()
                Loss.backward()
                optimizer_2.step()

                epoch_cost1ls.append(loss1)
                epoch_cost2ls.append(loss2)
                epoch_classifls.append(closs)
                epoch_loss.append(Loss)
                epoch_DGloss.append(DG_loss)

                y_true = yt.view(-1,1)
                y_true = y_true.cpu()

                y_trueDG = Lst.view(-1,1)
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

            # Take average across all training batches
            loss1tr_mean = l1[it]
            loss2tr_mean = l2[it]
            totlosstr_mean = L[it]
            DGlosstr_mean = DG_losstr[it]
            DGauctr_mean = DG_auctr[it]
            closstr_mean = classif[it]

            print("\n\nEpoch: {}".format(it))
            print("(train) loss1 mean: {}".format(loss1tr_mean))
            print("(train) loss2 mean: {}".format(loss2tr_mean))
            print("(train) total loss mean: {}".format(totlosstr_mean))
            print("################totlosstr_mean={}###############".format(totlosstr_mean))
            print("(train) DG loss mean: {}".format(DGlosstr_mean))

            # Write to file
            # Take avg
            save_model_to = SAVE_TRACE_TO + batch_sizes + '/' + model_params + '/model/'
            if not os.path.exists(save_model_to):
                os.makedirs(save_model_to)
                print("Directory ", save_model_to, " Created ")
#             else:
#                 print("Directory ", save_model_to, " already exists")
            save_best_model_to = os.path.join(save_model_to, split + '_best_model.pt')
            ## Saving multiple models in one file (Feature extractors, mapper, discriminators) ##
            ##  https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-multiple-models-in-one-file ##
            #torch.save({
            #            'Gen_state_dict': Gen.state_dict(),
            #            'Map_state_dict': Map.state_dict(),
            #            'DG_state_dict': DG.state_dict(),
            #            'optimizer_2_state_dict': optimizer_2.state_dict(),
            #            }, save_best_model_to)

        ## Evaluate model ##
        print("\n\n-- Evaluation -- \n\n")
        print("TXTestCells_N shape: {}\n".format(TXTestCells_N.size()))
        print("TYTestCells shape: {}\n".format(TYTestCells.size()))
        save_best_model_to = os.path.join(save_model_to, split + '_best_model.pt')
        test_loss, test_auc, test_apr, test_predicted_y = evaluate_model(TXTestCells_N, TYTestCells, Gen, Map)
        print("test_apr = {}\n".format(test_apr))
        print("\n\n-- Test Results --\n\n")
        print("test loss: {}".format(test_loss))
        print("test auc: {}".format(test_auc))
        print("\n ----------------- \n\n\n")
        with open(test_results_file, 'a') as f:
            f.write("-- Split {} --\n".format(split))
            f.write("Test loss: {}\t Test AUC: {}\t Test APR: {}\n\n\n".format(test_loss, test_auc, test_apr))
        AUCtest_splits_total.append(test_auc)
        APRtest_splits_total.append(test_apr)


                ## save test_predicted_y
        with open(test_predicted_y_file,'a') as p:
            TYTestCells = TYTestCells.cpu().detach().numpy()
            test_predicted_y = test_predicted_y.cpu().detach().numpy()
            for tp in range(TYTestCells.shape[0]):
                p.write("{}\t{}\t{}\t{}\n".format(split,YTestCells.index[tp],YTestCells['response'].values[tp],test_predicted_y[tp][0]))


        ##IntegratedGradients
        XIG = TXTestCells_N
        net = GP(Gen, Map)
        net = net.eval()
        out_probs = net(XIG).detach().cpu().numpy()

        ig = IntegratedGradients(net)

        XIG.requires_grad_()
        attr, delta = ig.attribute(XIG, target=0, return_convergence_delta=True, n_steps=100)
        attr = attr.cpu().detach().numpy()
        attr_ave = np.mean(attr, axis=0)
        if split == 'split1':
            ncbiID = XTrainGDSC.columns.values
            hgncID = ENTERZid2HGNCid(XTrainGDSC.columns)[0]
            attr_pd = pd.DataFrame({'NCBI':ncbiID,'HGNC':hgncID,split:attr_ave})
        else:
            attr_pd[split] = attr_ave
        if split == 'split5':
            col_ave = attr_pd[['split1','split2','split3','split4','split5']].mean(axis=1)
            attr_pd['average'] = col_ave


        if torch.equal(XIG, TXTestCells_N):
            ## save IG attribution values for target testing set
            attr_pd.to_csv(IG_target_test_file,sep="\t", index=False)

    ## Calculate Test set's avg AUC across different splits
    AUCtest_splits_total = np.array(AUCtest_splits_total)
    APRtest_splits_total = np.array(APRtest_splits_total)
    avgAUC = np.mean(AUCtest_splits_total)
    stdAUC = np.std(AUCtest_splits_total)
    avgAPR = np.mean(APRtest_splits_total)
    stdAPR = np.std(APRtest_splits_total)

    ####
    with open(test_results_file, 'a') as f:
        f.write("\n\n-- Average Test AUC --\n\n")
        f.write("Mean: {}\tStandard Deviation: {}\n".format(avgAUC, stdAUC))
        f.write("\n-- Average Test Precision Recall --\n\n")
        f.write("Mean: {}\tStandard Deviation: {}\n".format(avgAPR, stdAPR))

    with open(tr_ts_summary_path, "a") as f:
        f.write("Average Test AUC = {}; Average Test Precision Recall = {};  path = {}\n".format(avgAUC, avgAPR, test_results_file))
        f.close()
        ##

