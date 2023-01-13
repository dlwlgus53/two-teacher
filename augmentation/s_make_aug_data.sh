python make_aug_data.py \
    --short 0 \
    --labeled_data_path /home/jihyunlee/pptod/DST/inference_result/small/0.1/seed$1/train.json \
    --save_prefix ../data/seed$1_aug_data_$2 \
    --aug_model_path /home/jihyunlee/two-teacher/augmentation/model/teacher_aug$1_teacher_aug/model.pt \
    --update_number $2 \