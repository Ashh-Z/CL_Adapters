# Language Guidance in Vision Tasks

## Baseline
### Baseline Classification
```
python main_normal.py 
--experiment_id lg-ex-cif10-cls
--mode normal
--arch resnet18mam
--dataset cifar10
--lr 0.1
--batch_size 32
--n_epochs 100
--dataset_dir /volumes1/datasets/cifar/CIFAR10
--model er #not used
--scheduler cosine
```
### Baseline Continual Learning


## Explicit Language Guidance
### Classification
#### File : main_normal.py
mode -> vlm;

text_model, loss_mode, loss_wt, gpt_path
```
python main_normal.py 
--experiment_id lg-ex-cif10-cls
--mode vlm
--arch resnet18mam
--dataset cifar10
--lr 0.1
--batch_size 32
--n_epochs 100
--dataset_dir /volumes1/datasets/cifar/CIFAR10
--model er #not used
--scheduler cosine
--text_model sent_transf
--gpt_path /volumes1/datasets/cifar10_description.json
--loss_mode sim
--loss_wt 6 6 6 6
```

### Continual Learning
#### File : main.py

model -> vl_er; 

text_model, loss_mode, loss_wt, gpt_path
```
python main.py 
--experiment_id lg-ex-cif0-cil
--arch resnet18mam
--model vl_er
--dataset seq-cifar100
--buffer 200
--lr 0.03
--minibatch_size 32
--batch_size 32
--n_epochs 50
--dataset_dir /volumes1/datasets/cifar/CIFAR100
--ignore_other_metrics 1
--nowand 1
--text_model sent_transf
--gpt_path cl_datasets/metadata/cifar100_description.json
--loss_mode sim
--loss_wt 6 6 6 6
```

## Implicit Language Guidance

### Classification
#### File : main_normal.py
arch -> resnet18mamllm

mode -> normal, 

--llama; llm_block is clip or sent_transf, 
```
buffer_size = 200 (or 500)
python main_normal.py 
    --experiment_id lg-ix-cls \
    --arch resnet18mamllm \
    --mode normal \
    --seed 0 \
    --model vl_er \
    --dataset cifar100 \
    --lr 0.1 \
    --n_epochs 50 \
    --batch_size 32 \
    --minibatch_size 32 \
    --output_dir /output/ \
    --scheduler" "cosine" \
    --tensorboard \
    --llama \
    --llm_block clip \ #or sent_transf
```

### Continual Learning
#### File : main.py
arch -> resnet18mamllm

--llama; llm_block is clip or sent_transf, 
```
buffer_size = 200 (or 500)
python main.py 
    --experiment_id lg-ix-cl 
    --arch resnet18mamllm 
    --seed 0 
    --model vl_er
    --dataset seq-cifar100
    --buffer 200
    --lr 0.03
    --minibatch_size 32
    --batch_size 32
    --n_epochs 50
    --dataset_dir /volumes1/datasets/cifar/CIFAR100
    --ignore_other_metrics 1
    --nowand 1
    --text_model sent_transf
    --gpt_path cl_datasets/metadata/cifar100_description.json
    --loss_mode sim
    --loss_wt 6 6 6 6
    --llama 
    --llm_block clip  #or sent_transf
```


## Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

## Models

+ eXtended-DER (X-DER)

+ Dark Experience Replay (DER)
+ Dark Experience Replay++ (DER++)

+ Learning a Unified Classifier Incrementally via Rebalancing (LUCIR)
+ Greedy Sampler and Dumb Learner (GDumb)
+ Bias Correction (BiC)
+ Regular Polytope Classifier (RPC)

+ Gradient Episodic Memory (GEM)
+ A-GEM
+ A-GEM with Reservoir (A-GEM-R)
+ Experience Replay (ER)
+ Meta-Experience Replay (MER)
+ Function Distance Regularization (FDR)
+ Greedy gradient-based Sample Selection (GSS)
+ Hindsight Anchor Learning (HAL)
+ Incremental Classifier and Representation Learning (iCaRL)
+ online Elastic Weight Consolidation (oEWC)
+ Synaptic Intelligence
+ Learning without Forgetting
+ Progressive Neural Networks

## Datasets

**Class-Il / Task-IL settings**

+ Sequential MNIST
+ Sequential CIFAR-10
+ Sequential Tiny ImageNet
+ Sequential CIFAR-100

**Domain-IL settings**

+ Permuted MNIST
+ Rotated MNIST

**General Continual Learning setting**

+ MNIST-360

