#! /bin/bash

# --jobname, 作业在调度系统中的作业名
#SBATCH --job-name=unspervised_proj


# 申请使用gpu资源分区, --partition, 作业提交的制定分区
#SBATCH --partition=gpu 



# 申请一个节点，--nodes，申请节点数,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --output=nls21_sub_0235_%j.out


source ~/.bashrc
nvidia-smi
source /opt/app/anaconda3/bin/activate pytorch-1.12.1-cuda11.3.1-py38

python -u kmeans_nls6.py
