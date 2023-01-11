python make_aug_data.py \
    --short 1 \
    --labeled_data_path /home/jihyunlee/pptod/DST/inference_result/small/0.1/seed$1/train.json \
    --save_prefix ../data/debugging_aug_data_$1 \
    --aug_model_path /home/jihyunlee/two-teacher/augmentation/model/teacher_aug4_teacher_aug/epoch_7_loss_0.6595.pt \
    --scenario_percent $2 \