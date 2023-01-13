#!/bin/sh
#SBATCH -J teacher_aug # Job name
#SBATCH -o  ./out/teacher_aug.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A5000 # queue  name  or  partiton name

#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:2
#SBTACH   --ntasks=2
#SBATCH   --tasks-per-node=16
#SBATCH     --mail-user=jihyunlee@postech.ac.kr
#SBATCH     --mail-type=ALL

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## path  Erase because of the crash
module purge
module add cuda/10.4
module add cuDNN/cuda/10.4/8.0.4.30
#module  load  postech

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate QA_new "
conda activate QA_new

export PYTHONPATH=.



python learn_teacher_aug.py \
--short 0 \
--valid_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-dev.json \
--test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--labeled_data_path /home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_$1.json \
--max_epoch 20 \
--patient 5 \
--save_prefix ./model/teacher_v4_$1