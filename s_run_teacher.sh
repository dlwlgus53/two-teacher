

python learn_teacher.py \
--short 1 \
--valid_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-dev.json \
--labeled_data_path /home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_$1.json \
--test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--max_epoch 20 \
--save_prefix ./model/debugging_teacher$1

