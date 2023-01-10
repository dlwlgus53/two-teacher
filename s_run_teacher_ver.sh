
# python learn_teacher_ver.py \
# --short 0 \
# --valid_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-dev.json \
# --labeled_data_path /home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_$1.json \
# --test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
# --test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
# --verify_data_path /home/jihyunlee/pptod/data/multiwoz/data/unlabeled/0.1/unlabeled_$1.json \
# --max_epoch 20 \
# --save_prefix ./model/teacher_ver$1 



python learn_teacher_ver.py \
--short 0 \
--valid_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-dev.json \
--labeled_data_path /home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_$1.json \
--test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--verify_data_path /home/jihyunlee/pptod/data/multiwoz/data/unlabeled/0.1/unlabeled_$1.json \
--fine_trained /home/jihyunlee/two-teacher/model/model/teacher_ver1_teacher/epoch_11_loss_0.2172.pt \
--max_epoch 20 \
--save_prefix ./model/teacher_debug_ver$1 
