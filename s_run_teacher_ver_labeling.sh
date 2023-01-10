python learn_teacher_ver.py \
--short 0 \
--valid_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-dev.json \
--labeled_data_path /home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_$1.json \
--test_data_path /home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json \
--verify_data_path /home/jihyunlee/pptod/data/multiwoz/data/unlabeled/0.1/seed$1/multiwoz-fine-processed-test.json \
--fine_trained /home/jihyunlee/two-teacher/model/model/teacher_ver$1_teacher/model.pt \
--save_prefix ./model/teacher_ver_TF$1 