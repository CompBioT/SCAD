from sklearn import preprocessing

import os.path
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import GEOparse
from collections import Counter

import pandas as pd
import copy

##creat a Dict for GSE number and drug for matching

gsedrugs = {}
gsedrugs['Etoposide'] = 'GSE149215'
gsedrugs['PLX4720'] = 'GSE108383' ## default is A375 cell line
gsedrugs['PLX4720_451Lu'] = 'GSE108383' ## 451Lu cell line

DRUG = 'Etoposide'
GSEid = gsedrugs[DRUG]

Solid = "False"  ## use solid tumor cell lines, default is False
###
geneset=''
geneset="_ppi"
geneset="_tp4k"

os.chdir('D:/Project2/ANN/BMDSTL/BM')
SAVE_RESULT_TO = './data/split_norm/'+DRUG+'/'
LOAD_DATA_FROM = './data/original_norm/'+DRUG+'/'    # where the original (pre-split) data is stored
print(LOAD_DATA_FROM)

dirName = SAVE_RESULT_TO
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")

##GDSC drug log IC50 and binarized repsonse
## reference script from https://github.com/hosseinshn/MOLI/blob/master/supplementary_and_QC/drug_responses.ipynb
## https://github.com/hosseinshn/MOLI/blob/master/preprocessing_scr/annotations.ipynb
## https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources///Data/suppData/TableS5C.xlsx
## https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/suppData/TableS1E.xlsx
df = pd.read_csv("./data/response/GDSC_response.all_drugs.tsv",sep='\t', index_col=0, decimal='.')
#### extract all drugs log(IC50) information ####
df_ic50 = pd.read_csv("./data/response/GDSC_response.logIC50.all_drugs.tsv",sep='\t', index_col=0, decimal='.')

###get DRUG log IC50 and binarized response
DRUG_name = DRUG
if DRUG_name == "PLX4720_451Lu":
    DRUG_name = "PLX4720"

d1 = df_ic50.loc[:,[DRUG_name]]
d1.columns = ["logIC50"]
d2 = df.loc[:,[DRUG_name]]
d2.columns = ["response"]
print("original cell line count with logIC50 = {}".format(len(d1.index)))
print("original cell line count with binarized label = {}".format(len(d2.index)))

d1.dropna(inplace=True)
d2.dropna(inplace=True)

####decide whether to use solid tumor cell lines only####
COSMIC_ids = pd.read_excel("./data/response/TableS1E.xlsx")
COSMIC_ids = COSMIC_ids.iloc[3:,[1,2,8]]
COSMIC_ids = COSMIC_ids.iloc[:-1,]
COSMIC_ids.columns = ["name",'COSMIC','Tissue_descriptor1']
COSMIC_ids.set_index("name", inplace=True, drop=True)   ## 1001 * 2

solid_tumors = COSMIC_ids['COSMIC'][(COSMIC_ids['Tissue_descriptor1'] != "leukemia") &
                     (COSMIC_ids['Tissue_descriptor1'] != "lymphoma") &
                     (COSMIC_ids['Tissue_descriptor1'] != "myeloma")
                    ]

if Solid == "True":
    d1 = d1[d1.index.isin(solid_tumors)]
    d2 = d2[d2.index.isin(solid_tumors)]

Counter(d1.index == d2.index)

####decide whether to use solid tumor cell lines only####

print("intersect cell line count with logIC50 = {}".format(len(d1.index)))
print("intersect cell line count with binarized label = {}".format(len(d2.index)))
##check ids
Counter(d1.index == d2.index)
d_drug = pd.concat([d2, d1],axis=1)
d_drug.loc[:, "drug"] = DRUG
d_drug.index.name = "sample_name"
d_drug.to_csv("./data/response/"+'GDSC_response'+"_"+DRUG+".tsv",sep = "\t")
d_drug.shape

### prepare GDSC Drug specific microarray RMA expression data ###
# gdsc-rma_gene-expression.csv downloaded from https://ibm.ent.box.com/v/paccmann-pytoda-data/folder/91948853171
gdsc_rma_all = pd.read_csv("./data/original/gdsc-rma_gene-expression.csv",sep=',', index_col=0, decimal='.')
gdsc_rma_all_cp = gdsc_rma_all.copy()

#gdsc_rma_all.index.name = "cell_line"
#gdsc_rma_all.columns = genes
def HGNCid2ENTERZid(genelist):
    # self.dict_ens = {}
    dict_sym = {}
    HGNC = pd.read_csv("./data/original/HGNC_symbol_all_genes.tsv", sep='\t', index_col=0, decimal='.')
    for index in range(len(HGNC['Approved symbol'])):
        if (not isinstance(HGNC['Approved symbol'][index], float)):  ##check Approved symbol is not empty
            if HGNC['NCBI Gene ID'][index] == HGNC['NCBI Gene ID'][index]:
                dict_sym[HGNC['Approved symbol'][index]] = int(HGNC['NCBI Gene ID'][index])
    #
    ncbilist = []
    indexkeep = []

    for index in range(len(genelist)):
        if genelist[index] in dict_sym:
            indexkeep.append(index)
            ncbilist.append(dict_sym[genelist[index]])

    return (ncbilist,indexkeep)

etz = HGNCid2ENTERZid(gdsc_rma_all.columns)

ncbilist = etz[0]
indexkeep = etz[1]
#
hgnckeep = gdsc_rma_all.columns[indexkeep]

gdsc_rma_all.shape
gdsc_rma_all = gdsc_rma_all.iloc[:,indexkeep]
gdsc_rma_all.columns = ncbilist

########### Get intersect DRUG cell lines with both rma and drug DRUG IC50 value
drug_cells = []
for i in (gdsc_rma_all.index):
    if i in d_drug.index:
        drug_cells.append(i)

drug_cells.sort()  ## sort by cell line name

GDSC_exprs_drug = gdsc_rma_all.loc[drug_cells]
print("GDSC Sample Size: {}".format(GDSC_exprs_drug.shape))

d_drug_intersect = d_drug.loc[drug_cells]
Counter(GDSC_exprs_drug.index == d_drug_intersect.index) #
print("GDSC cell line index checking: {}".format(Counter(GDSC_exprs_drug.index == d_drug_intersect.index)))  #(875, 15299)

d_drug_intersect['logIC50'].describe()

print("GDSC: Num of resistant and Num of Sensitive = {}".format(Counter(d_drug_intersect['response'].values)))
d_drug_intersect['response'].values[d_drug_intersect['response'].values == 'R'] = 0       ## 0 means resistant
d_drug_intersect['response'].values[d_drug_intersect['response'].values == 'S'] = 1       ## 1 means sensitive
print("GDSC: Num of resistant and Num of Sensitive = {}".format(Counter(d_drug_intersect['response'].values)))

######save unscaled GDSC expression data, prepared for differential expression analysis######
gdsc_exprs_drug_RMA_save_path=os.path.join(SAVE_RESULT_TO, 'Source_exprs_resp_RMA.'+DRUG+geneset+'.tsv')
GDSC_exprs_drug_rma = copy.deepcopy(GDSC_exprs_drug)
GDSC_exprs_drug_rma.columns = hgnckeep
GDSC_exprs_drug_rma.insert(0, "response", list(d_drug_intersect['response']), allow_duplicates=False)
GDSC_exprs_drug_rma.insert(1, "logIC50", list(d_drug_intersect['logIC50']), allow_duplicates=False)
GDSC_exprs_drug_rma.to_csv(path_or_buf=gdsc_exprs_drug_RMA_save_path,
                            sep='\t', index=True)
##############################################
####
from statistics import *

median(d_drug_intersect['logIC50'].values)

scaler1 = StandardScaler()

GDSC_exprs_drug[GDSC_exprs_drug.columns] = scaler1.fit_transform(GDSC_exprs_drug[GDSC_exprs_drug.columns])

GDSC_exprs_drug_z = GDSC_exprs_drug

#######################################################################
##      data preprocessing for GSE108383 expr.z.resp tsv table       ##
#######################################################################
# Loading Normalized data:
GSE_exprs_z = pd.read_csv(LOAD_DATA_FROM+'exprs/'+GSEid+'_exprs.z.'+DRUG+geneset+'.tsv',
                                     sep='\t', index_col=0, decimal='.')

## append drug response label for DIFFERENT GSE dataset, each dataset require customized step ##
GSE_response = []

GSE_exprs_z.index #row names

if DRUG.startswith('PLX4720') and GSEid == 'GSE108383':
    for samp in GSE_exprs_z.columns:
        response = []
        if "_sc_br_" in samp:
            response = 0  ## 0 means resistant
        elif "_par_" in samp:
            response = 1  ## 1 means sensitive
        else:
            raise NotImplementedError('DRUG {} and GSEid {} not match'.format(DRUG, GSEid))
        GSE_response.append(response)
elif DRUG == 'Etoposide' and GSEid == 'GSE149215':
    for samp in GSE_exprs_z.columns:
        response = []
        if "PC9D0C" in samp:
            response = 1  ## 1 means sensitive
        elif "PC9Day3" in samp:
            response = 0  ## 0 means resistant
        else:
            raise NotImplementedError('samp {} and response {} not match'.format(samp, response))
        GSE_response.append(response)

else:
    raise NotImplementedError('DRUG {} and GSEid {} not match'.format(DRUG, GSEid))

# count resistant and sensitive cells
print("{}: Num of resistant(0) and Num of Sensitive(1) = {}".format(GSEid, Counter(GSE_response)))
# Transpose the pandas data frame for expression so that the columns = genes (features) and rows = samples (cell ids)
GSE_exprs_z = pd.DataFrame.transpose(GSE_exprs_z)
GDSC_exprs_drug_z = GDSC_exprs_drug

ls = GSE_exprs_z.columns.intersection(GDSC_exprs_drug.columns)
GDSC_exprs_drug_z = GDSC_exprs_drug_z.loc[:,ls]
GSE_exprs_z = GSE_exprs_z.loc[:,ls]

print("GDSC sample size and number of genes = {}".format(GDSC_exprs_drug_z.shape))
print("{} sample size and number of genes = {}".format(GSEid, GSE_exprs_z.shape))

# check overlap genes
GDSC_exprs_drug_z_genes = list(GDSC_exprs_drug_z.columns.values)
GSE_exprs_z_genes = list(GSE_exprs_z.columns.values)

if not (GDSC_exprs_drug_z_genes == GSE_exprs_z_genes):
    print("\nWARNING: genes do not have the same order\n")

## Add drug response binary value and logIC50 value to GDSC_exprs_drug_z dataframe
GDSC_exprs_drug_z_resp = GDSC_exprs_drug_z.copy(deep=True)
GDSC_exprs_drug_z_resp.insert(0, "response", list(d_drug_intersect['response']), allow_duplicates=False)
GDSC_exprs_drug_z_resp.insert(1, "logIC50", list(d_drug_intersect['logIC50']), allow_duplicates=False)

GDSC_exprs_drug_z_resp = shuffle(GDSC_exprs_drug_z_resp, random_state=42)

##save GDSC source file
gdsc_save_path=os.path.join(SAVE_RESULT_TO, 'Source_exprs_resp_z.'+DRUG+geneset+'.tsv')
if Solid=="True":
    gdsc_save_path = os.path.join(SAVE_RESULT_TO, 'Source_solid_exprs_resp_z.' + DRUG + geneset + '.tsv')

GDSC_exprs_drug_z_resp.to_csv(path_or_buf=gdsc_save_path,
                            sep='\t', index=True)

print(" - - successfully saved GDSC {} Source data - - \n".format(DRUG))

GSE_exprs_z_resp = GSE_exprs_z.copy(deep=True)
GSE_exprs_z_resp.insert(0,"response",GSE_response, allow_duplicates=False)

GSE_exprs_z_resp = shuffle(GSE_exprs_z_resp, random_state=42)
GSE_exprs_z_resp.to_csv(path_or_buf=os.path.join(SAVE_RESULT_TO, 'Target_expr_resp_z.'+DRUG+geneset+'.tsv'),
                            sep='\t', index=True)
print(" - - successfully saved preprocessed {} {} Target data - - \n".format(GSEid, DRUG))