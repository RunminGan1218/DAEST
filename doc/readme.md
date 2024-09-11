### config the parameter and proj/exp in the cfgs/config.yaml
    config the dataset in data cfgs/folder
    config the model parameters in cfgs/model folder
    config the training setting in cfgs/config.yaml file (working dir/cp saving dir)



### run python file(You can run all these steps at once using the .sh file in the scripts folder.)
1. run train_ext.py  pretrain 
2. run extract_fea.py   running norm lds save
3. run train_mlp.py   finetune

#### caution:
To ensure code security, please ensure that the same run_no is used when running three code files within a trial.


### use wandb to see results





