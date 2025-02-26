#!/bin/bash
# Define the datasets you want to run
datasets=("seq-cifar100")#" "seq-cifar100")

declare -A data_path_lst
data_path_lst["seq-cifar10"]='/home/snarasimhe/workspace/datasets/CIFAR10'
data_path_lst["seq-cifar100"]='/home/snarasimhe/workspace/datasets/CIFAR100'

# Define the models and their parameters
declare -A model_params
model_params["vl_er"]="--buffer_size 200 --minibatch_size 32"

# Define the list of buffer sizes
buffer_sizes=(200 500)
tasks_cif100_lst=(5 10)
lst_lr=(0.1)  # Learning rate list
num_runs=2
start_seed=42
tasks_cif100=(5 10)

# Loop over all combinations
for dataset in "${datasets[@]}"; do
  for buffer_size in "${buffer_sizes[@]}"; do
    for model in "${!model_params[@]}"; do
      for lr in "${lst_lr[@]}"; do
        for task in "${tasks_cif100[@]}"; do
          for seed in $(seq $start_seed $((start_seed + num_runs - 1))); do
            exp_id="${model}-${dataset}-t${task}-b${buffer_size}-lr-${lr}-s-${seed}"
            echo "Submitting job for combination: $exp_id"
            # Create a temporary script file
            tmp_script=$(mktemp /home/snarasimhe/workspace/continual_VLM/tmp/slurm_script.XXXXXX)
            cat <<EOF > "$tmp_script"
#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:45:00
#SBATCH --cpus-per-task=18
#SBATCH -o /home/snarasimhe/workspace/continual_VLM/slurm/${dataset}/slurm-%j.out
#SBATCH -e /home/snarasimhe/workspace/continual_VLM/slurm/${dataset}/slurm-%j.err

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_cl

# Run the Python script with the current parameters
/home/snarasimhe/miniconda3/envs/vlm_cl/bin/python3 main.py \
  --experiment_id $exp_id \
  --model $model \
  --arch resnet18mam \
  --dataset $dataset \
  --dataset_dir ${data_path_lst[$dataset]} \
  --lr $lr \
  --n_epochs 50 \
  --batch_size 32 \
  --tensorboard 0 \
  --nowand 1 \
  --ignore_other_metrics 1 \
  --wandb_project continual_VLM \
  --wandb_entity sngowda42 \
  --output_dir /home/snarasimhe/workspace/continual_VLM/results_final \
  --seed $seed \
  --buffer_size $buffer_size \
  --minibatch_size 32 \
  --n_tasks_cif task
EOF
            # Submit the temporary script
            sbatch "$tmp_script"
            rm "$tmp_script"
          done
        done
      done
    done
  done
done
