# SCAD

python SCAD/data/split_norm/split_data_SCAD_5fold_norm.py

python SCAD/model/SCAD_train_binarized_5folds.py  -e FX -d Gefitinib -g _norm -s 42 -h_dim 1024 -z_dim 256 -ep 20 -la1 0.2 -mbS 32 -mbT 32
