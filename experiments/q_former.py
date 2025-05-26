import sys

sys.path.append(".")

import torch
from modules.q_former.q_former import BertLMHeadModel, BertModel, BertConfig
from transformers.models.bert import BertLMHeadModel as BertLMHeadModelFromHF

config = BertConfig.from_pretrained(
    "dbmdz/bert-base-german-europeana-cased",
)
config.is_decoder = True
shared_encoder = BertLMHeadModel(config)
bt = BertLMHeadModelFromHF.from_pretrained(
    "dbmdz/bert-base-german-europeana-cased",
    config=config,
    device_map="cpu",
    torch_dtype=torch.float32,
).state_dict()
shared_encoder.load_state_dict(bt, strict=False)
shared_encoder.cuda()
for name, p in shared_encoder.named_parameters():
    print(name)
