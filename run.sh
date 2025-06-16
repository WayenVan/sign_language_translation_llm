# python3 ./scripts/train.py -cn initial_train_8a100 modules.visual_adapter.num_extra_queries=8
# python3 ./scripts/train.py -cn initial_train_8a100 modules.visual_adapter.num_extra_queries=16
#
#
# python3 ./scripts/train.py -cn initial_train_8a100 modules.num_query_tokens=64
# python3 ./scripts/train.py -cn initial_train_8a100 modules.num_query_tokens=128
# python3 ./scripts/train.py -cn initial_train_8a100 modules.visual_adapter.num_extra_queries=16
# python3 ./scripts/train.py -cn initial_train_8a100 vtm_flag=False vtc_flag=False
# python3 ./scripts/train.py -cn initial_train_8a100 vtc_flag=False
python3 ./scripts/train.py -cn initial_train_8a100
# python3 ./scripts/train.py -cn initial_train_8a100 modules.num_query_token=128
