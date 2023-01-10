

python learn_teacher.py \
--short 0 \
--valid_data_path /home/jihyunlee/woz-data/MultiWOZ_2.1/dev_data.json \
--labeled_data_path /home/jihyunlee/woz-data/MultiWOZ_2.1/labeled/0.1/labeled_$1.json \
--test_data_path /home/jihyunlee/woz-data/MultiWOZ_2.1/test_data.json \
--max_epoch 10 \
--save_prefix ./model/debugging_teacher$1