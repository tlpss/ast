#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./log_%j.txt

# set -x
# # comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source 
# export TORCH_HOME=../../pretrained_models


# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --sensor_type) sensor_type="$2"; shift ;;
    --dataset-dir) dataset_dir="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Default values if not provided
sensor_type=${sensor_type:-mic}
dataset_dir=${dataset_dir:-./data/icra2025-v1}

echo "Sensor type: ${sensor_type}"
echo "Dataset directory: ${dataset_dir}"

# set this to the mean and std of the dataset for the selected sensor type!
# python ../../src/get_norm_stats.py --datafile data/icra2025-v0/robomic_all_mic.json --label_csv data/robomic_categories.csv 
if [ ${sensor_type} == 'mic' ]
then
  dataset_mean=-4.99
  dataset_std=2.815

else
  # laser data statistics
  dataset_mean=-1.17
  dataset_std=2.42
fi

dataset=robomic-${sensor_type}
model=ast

# imagenetpretrain=False
# audiosetpretrain=False
# num_mel_bins=64

imagenetpretrain=True
audiosetpretrain=True
num_mel_bins=128

bal=none
if [ $audiosetpretrain == True ]
then
  lr=5e-6
else
  lr=1e-4
fi
freqm=48
timem=10
mixup=0.5
epoch=10
batch_size=4
fstride=10
tstride=10

audio_length=1100 # 11s
noise=False

metrics=acc
loss=CE
warmup=False
lrscheduler_start=3
lrscheduler_step=1
lrscheduler_decay=0.85

base_exp_dir=./exp/test-${dataset}-$(date +%Y%m%d_%H%M%S)

# if [ -d $base_exp_dir ]; then
#   echo 'exp exist'
#   exit
# fi
mkdir -p $base_exp_dir

for((fold=0;fold<=4;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=${dataset_dir}/robomic_train_${sensor_type}_fold_${fold}.json
  te_data=${dataset_dir}/robomic_val_${sensor_type}_fold_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/robomic_categories.csv --n_class 2 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
  --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} --seed $((2025+${fold}))
done

python ./get_robomic_result.py --exp_path ${base_exp_dir}