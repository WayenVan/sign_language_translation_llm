import numpy as np
from nlpaug.augmenter.word.random import RandomWordAug


text = "Das Wetter ist heute sehr schön, wir gehen im Park spazieren."

# 创建swap增强器，默认会随机交换部分词语位置（局部调换）
aug = RandomWordAug(action="swap")

augmented_text = aug.augment(text)
print(augmented_text)
