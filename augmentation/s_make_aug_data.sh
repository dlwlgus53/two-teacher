
python make_aug_data.py \
    --short 0 \
    --labeled_data_path /home/jihyunlee/woz-data/MultiWOZ_2.1/labeled/0.01/labeled_$1.json \
    --save_prefix ../data/debugging_aug_data_$1 \
    --aug_model_path /home/jihyunlee/two-teacher/model/model/debugging_teacher_aug1_teacher_aug/epoch_4_loss_0.6575.pt \
    --scenario_percent 0.1 \
    --aug_percent 0.1 \