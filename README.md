# SCAD

## data splitting
python SCAD/data/split_norm/split_data_SCAD_5fold_norm.py

## hyper-parameter selection for each drug
python SCAD/model/SCAD_5foldsCV_on_source_arg.py -e FX -d Gefitinib -g _norm -adj N

## retrain the model with selected hyper-parameters and test on target domain scRNA-seq data
### the hyper-parameters for our best model (weight_all: under weight sampling and considering all genes) are kept in Supplementary Table S5 in our manuscript
### example usage:
python SCAD/model/SCAD_train_binarized_5folds.py -e FX -d Gefitinib -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 20 -la1 0.2 -mbS 32 -mbT 32
