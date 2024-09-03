python preprocess.py
python train_FCMSAN.py -c config/ISRUC.config -g 0,1
python train_MKSTGCN.py -c ./config/ISRUC.config -g 0,1
python evaluate_MSTGCN.py -c ./config/ISRUC.config -g 1
