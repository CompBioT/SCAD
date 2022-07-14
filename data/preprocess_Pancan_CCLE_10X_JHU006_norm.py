import os.path
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from collections import Counter
import copy

import pandas as pd

MODEL = 'BMDSTL'
os.chdir('D:/Project2/ANN/BMDSTL/BM')

drugls = ['Gefitinib','Vorinostat','AR-42']
DRUG ='Gefitinib'
cellline = 'jhu006'
Suffix = ['scrna_ccle']

Solid = "False" ## use solid tumor cell lines, default is False

###
geneset=''
#geneset="_ppi"
#geneset="_tp4k"

SAVE_RESULT_TO = './data/split_norm/'+DRUG+'/'
LOAD_DATA_FROM = './data/original_norm/Pancan_CCLE_10X/'    # where the original (pre-split) data is stored
print(LOAD_DATA_FROM)

dirName = SAVE_RESULT_TO
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")

##GDSC drug log IC50 and binarized repsonse
## reference script from https://github.com/hosseinshn/MOLI/blob/master/supplementary_and_QC/drug_responses.ipynb
#D:\Project2\ANN\BMDSTL\BM\data\response\TableS1E.xlsx
df = pd.read_csv("./data/response/GDSC_response.all_drugs.tsv",sep='\t', index_col=0, decimal='.')
#### extract all drugs log(IC50) information ####
df_ic50 = pd.read_csv("./data/response/GDSC_response.logIC50.all_drugs.tsv",sep='\t', index_col=0, decimal='.')

###get DRUG log IC50 and binarized response
d1 = df_ic50.loc[:,[DRUG]]
d1.columns = ["logIC50"]
d2 = df.loc[:,[DRUG]]
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
COSMIC_ids.set_index("name",inplace=True,drop=True)   ## 1001 * 2

solid_tumors = COSMIC_ids['COSMIC'][(COSMIC_ids['Tissue_descriptor1'] != "leukemia") &
                     (COSMIC_ids['Tissue_descriptor1'] != "lymphoma") &
                     (COSMIC_ids['Tissue_descriptor1'] != "myeloma")
                    ]
if Solid == "True":
    d1=d1[d1.index.isin(solid_tumors)]
    d2=d2[d2.index.isin(solid_tumors)]

Counter(d1.index==d2.index)

####decide whether to use solid tumor cell lines only####

print("intersect cell line count with logIC50 = {}".format(len(d1.index)))
print("intersect cell line count with binarized label = {}".format(len(d2.index)))
##check ids
Counter(d1.index == d2.index)
d_drug = pd.concat([d2,d1],axis=1)
d_drug.loc[:,"drug"] = DRUG
d_drug.index.name = "sample_name"
d_drug.to_csv("./data/response/"+Suffix[0]+"_"+DRUG+".tsv",sep = "\t")
d_drug.shape  # (896,3) for Belinostat;

### prepare GDSC ['Belinostat','NU-7441','Cisplatin'] microarray RMA expression data ###
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
        if (not isinstance(HGNC['Approved symbol'][index], float)):
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
print("GDSC Sample Size: {}".format(GDSC_exprs_drug.shape))  #(875, 15299)

d_drug_intersect = d_drug.loc[drug_cells]
Counter(GDSC_exprs_drug.index == d_drug_intersect.index) #Counter({True: 875})
print("GDSC cell line index checking: {}".format(Counter(GDSC_exprs_drug.index == d_drug_intersect.index)))  #(875, 15299)

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
#############################################################################################

scaler1 = StandardScaler()

GDSC_exprs_drug[GDSC_exprs_drug.columns] = scaler1.fit_transform(GDSC_exprs_drug[GDSC_exprs_drug.columns])

GDSC_exprs_drug_z = GDSC_exprs_drug

#######################################################################
##      data preprocessing for ccle rnaseq hnscc cell line jhu006/scc47 expr.z.resp tsv table       ##
#######################################################################

## Read metadata ##
Cell_line = []
if cellline == 'jhu006':
    Cell_line = 'JHU006_UPPER_AERODIGESTIVE_TRACT'
else:
    raise NotImplementedError('{} is not a valid extractor'.format(Cell_line))

pancan_sc_meta = pd.read_csv(LOAD_DATA_FROM+"Metadata.txt",sep='\t', index_col=0, decimal='.')
pancan_sc_meta.drop(['TYPE'])
hnscc_sc_meta = pancan_sc_meta[pancan_sc_meta['Cell_line'] == Cell_line]
hnscc_sc_meta = hnscc_sc_meta[["Cell_line", "EpiSen_score"]]
hnscc_sc_meta["EpiSen_score"] = pd.to_numeric(hnscc_sc_meta["EpiSen_score"])
hnscc_sc_meta = hnscc_sc_meta.sort_values(by=['EpiSen_score'])

#correct one
cells_EpiSen_low = hnscc_sc_meta.index[hnscc_sc_meta['EpiSen_score'] <= hnscc_sc_meta['EpiSen_score'].quantile(0.1)]
cells_EpiSen_high = hnscc_sc_meta.index[hnscc_sc_meta['EpiSen_score'] > hnscc_sc_meta['EpiSen_score'].quantile(0.9)]

EpiSen_select = []
EpiSen_select.extend(cells_EpiSen_low)
EpiSen_select.extend(cells_EpiSen_high)

hnscc_exprs_z = pd.read_csv(LOAD_DATA_FROM+Suffix[0]+'_'+cellline+'_exprs.z'+geneset+'.tsv',
                                     sep='\t', index_col=0, decimal='.')

hnscc_exprs_z = hnscc_exprs_z[EpiSen_select]

# append drug sensitivity for cell lines jhu006 and scc47
hnscc_samples = hnscc_exprs_z.columns

cell_sensitivity = []

if DRUG == 'Gefitinib':
    for samp in hnscc_exprs_z.columns:
        sens = []
        if samp in cells_EpiSen_high:
            sens = 1      ## 1 means sensitive
        elif samp in cells_EpiSen_low:
            sens = 0      ## 0 means resistant/not sensitive
        cell_sensitivity.append(sens)
elif DRUG == 'Vorinostat' or DRUG == 'AR-42':
    for samp in hnscc_exprs_z.columns:
        sens = []
        if samp in cells_EpiSen_low:
            sens = 1  ## 1 means sensitive
        elif samp in cells_EpiSen_high:
            sens = 0  ## 0 means resistant/not sensitive
        cell_sensitivity.append(sens)
else:
    raise NotImplementedError('{} is not a valid drug name'.format(DRUG))

# count resistant and sensitive cells
print("JHU006: Num of resistant and Num of Sensitive = {}".format(Counter(cell_sensitivity)))

hnscc_exprs_z = pd.DataFrame.transpose(hnscc_exprs_z)

ls = hnscc_exprs_z.columns.intersection(GDSC_exprs_drug.columns)
len(ls)  #10684
GDSC_exprs_drug_z = GDSC_exprs_drug.loc[:,ls]
hnscc_exprs_z = hnscc_exprs_z.loc[:,ls]

print("GDSC sample size and number of genes = {}".format(GDSC_exprs_drug_z.shape))
print("JHU006 sample size and number of genes = {}".format(hnscc_exprs_z.shape))

# check overlap genes
GDSC_drug_exprs_z_genes = list(GDSC_exprs_drug_z.columns.values)
hnscc_exprs_z_genes = list(hnscc_exprs_z.columns.values)

if not (GDSC_drug_exprs_z_genes == hnscc_exprs_z_genes):
    print("\nWARNING: genes do not have the same order\n")

## Add drug response binary value and logIC50 value to GDSC_exprs_drug_z dataframe
GDSC_exprs_drug_z_resp = GDSC_exprs_drug_z.copy(deep=True)
GDSC_exprs_drug_z_resp.insert(0, "response", list(d_drug_intersect['response']), allow_duplicates=False)
GDSC_exprs_drug_z_resp.insert(1, "logIC50", list(d_drug_intersect['logIC50']), allow_duplicates=False)

gdsc_save_path=os.path.join(SAVE_RESULT_TO, 'Source_exprs_resp_z.'+DRUG+geneset+'.tsv')

if Solid=="True":
    gdsc_save_path = os.path.join(SAVE_RESULT_TO, 'Source_solid_exprs_resp_z.' + DRUG + geneset + '.tsv')

GDSC_exprs_drug_z_resp.to_csv(path_or_buf=gdsc_save_path,
                            sep='\t', index=True)

print(" - - successfully saved GDSC drug Source data - - \n")

hnscc_exprs_z_resp = hnscc_exprs_z.copy(deep=True)

hnscc_exprs_z_resp.insert(0,"response",cell_sensitivity, allow_duplicates=False)
hnscc_exprs_z_resp = shuffle(hnscc_exprs_z_resp, random_state=50)

hnscc_exprs_z_resp.to_csv(path_or_buf=os.path.join(SAVE_RESULT_TO, 'Target_expr_resp_z.'+DRUG+geneset+'.tsv'),
                            sep='\t', index=True)
print(" - - successfully saved preprocessed hnscc Target data - - \n")
