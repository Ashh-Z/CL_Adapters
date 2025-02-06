CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python main.py \
--arch resnet18 \
--model sgd \
--dataset seq-cifar10 \
--dataset_dir volumes1/datasets/cifar/CIFAR10 \
--load_best_args \
--nowand 1 \
--use_adapter \
--multihead \
--pretrained \
--freeze_backbone