import json

data = []

with open('data/java/train.jsonl', 'r') as f:
    for line in f:
        js = json.loads(line)
        data.append(js)
import random

test01 = random.sample(data, 12288)
with open('data/java/test02.jsonl', 'w') as f:
      # 遍历列表中的每个元素
      for item in test01:
          # 将元素转换为JSON格式字符串
          json_string = json.dumps(item)
          # 将JSON字符串写入文件，并在行末添加换行符
          f.write(json_string + '\n')  