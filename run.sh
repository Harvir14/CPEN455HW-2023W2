python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 50 \
--sampling_interval 25 \
--save_interval 50 \
--dataset cpen455 \
--nr_resnet 1 \
--nr_filters 10 \
--nr_logistic_mix 5 \
--lr_decay 0.999995 \
--max_epochs 50 \
--en_wandb True \
