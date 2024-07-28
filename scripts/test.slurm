#!/bin/bash

#SBATCH --job-name=727_node1test   # 作业名
#SBATCH --partition=gpu            # 分区名称
#SBATCH -n 1                       # 任务数
#SBATCH --ntasks-per-node=1        # 每个节点任务数
#SBATCH --gres=gpu:1               # 使用一个GPU
#SBATCH --nodelist=node1           # 指定节点
#SBATCH --mail-type=end
#SBATCH --mail-user=sphere012@outlook.com
#SBATCH --output=727_node1_test.out
#SBATCH --error=727_node1_test.err
# 完成配置

# 激活虚拟环境
source /public/home/ymhu/anaconda3/bin/activate        # 替换为你的虚拟环境路径

# 打印虚拟环境信息
which python
python --version
