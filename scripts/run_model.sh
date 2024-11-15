run=1
dataset=FACED_def   #set your dataset and config the path in corresponding yaml file in data folder
gpus='[0]'
valid_method='loo'  # 'loo' or int number
logging=default
iftest=False
msFilter_timeLen=3
timeFilterLen=30
dilation_array='[1,3,6,12]'
seg_att=15
avgPoolLen=15
timeSmootherLen=3
activ='sigmoid'
model=cnn_att
has_att=True
global_att=False
ext_wd=0.00015
mlp_wds=(0.0022 0.005)

iftest=True   # if true, only run 1 fold


proj_name="$dataset""_epoch30""_$model""_att$has_att""_global$global_att""_act$activ"

exp_name="$model""_restart$restart_times"


# you can run it in once
echo "proj: $proj_name exp:$exp_name run: $run gpus: $gpus valid_method: $valid_method"
echo "train ext wd: $ext_wd"
python train_ext.py log.run=$run log.proj_name=$proj_name data=$dataset model=$model\
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name  \
                    train.max_epochs=30 train.min_epochs=5 train.patience=30\
                    model.msFilter_timeLen=$msFilter_timeLen\
                    model.dilation_array=$dilation_array \
                    model.seg_att=$seg_att model.avgPoolLen=$avgPoolLen model.timeSmootherLen=$timeSmootherLen\
                    model.timeFilterLen=$timeFilterLen\
                    model.activ=$activ model.has_att=$has_att model.global_att=$global_att\
                    train.iftest=$iftest
                    

                    
echo "extract fea with wd: $ext_wd"
python extract_fea.py log.run=$run log.proj_name=$proj_name data=$dataset \
                      train.gpus=$gpus train.valid_method=$valid_method \
                      hydra/job_logging=$logging train.iftest=$iftest \
                      log.exp_name=$exp_name \
                      train.iftest=$iftest\
                      # ext_fea.mode='de'


for mlp_wd in "${mlp_wds[@]}"
do
  echo "exp_name: $exp_name"
  echo "Training MLP with mlp_wd: $mlp_wd and ext_wd: $ext_wd"
  python train_mlp.py log.run=$run log.proj_name=$proj_name data=$dataset \
                    train.gpus=$gpus train.valid_method=$valid_method \
                    hydra/job_logging=$logging train.iftest=$iftest \
                    log.exp_name=$exp_name \
                    mlp.wd=$mlp_wd \
                    train.iftest=$iftest\
                    # ext_fea.mode='de'

done