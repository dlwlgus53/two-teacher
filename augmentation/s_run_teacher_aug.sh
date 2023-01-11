python learn_teacher_aug.py \
--short 0 \
--valid_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-dev.json \
--test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--labeled_data_path /home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_$1.json \
--max_epoch 20 \
--save_prefix ../model/teacher_aug$1