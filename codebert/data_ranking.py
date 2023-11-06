import bleu
import math
import random
import json

output_path = "codebert/saved_models/java/test.output"
gold_path ="codebert/saved_models/java/test.gold"

pred = []
gold = []


with open(output_path, "r") as file:
    for line in file:
        pred.append(line.strip())
for i in range(len(pred)):
    pred[i]= pred[i].split('	',1)[1]
with open(gold_path, "r") as file:
    for line in file:
        gold.append(line.strip())
for i in range(len(gold)):
    gold[i]= gold[i].split('	',1)[1]

scores = []
 
for i in range(len(pred)):
    
    predictions =  pred[i]
    gold_fn = gold[i] 
    pre = []
    truth = []
    pre.append(str(0) + '\t' + predictions)
    truth.append(str(0) + '\t' + gold_fn)          
    (goldMap, predictionMap) = bleu.computeMaps_train(pre, truth)
    bleuscore = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    scores.append(bleuscore)



def quickselect(nums, k):
    if len(nums) == 1:
        return nums[0]
    pivot = random.choice(nums)
    lows = [n for n in nums if n < pivot]
    highs = [n for n in nums if n > pivot]
    pivots = [n for n in nums if n == pivot]
    if k <= len(highs):
        return quickselect(highs, k)
    elif k <= len(highs) + len(pivots):
        return pivots[0]
    else:
        return quickselect(lows, k - len(highs) - len(pivots))
threshold1 = quickselect(scores,12288*1/4)


test01 = []
with open('data/java/test02.jsonl', 'r') as f:
    for line in f:
        js = json.loads(line)
        test01.append(js)

train01 = [] 
for i in range(len(test01)):
    predictions =  pred[i]
    gold_fn = gold[i] 
    pre = []
    truth = []
    pre.append(str(0) + '\t' + predictions)
    truth.append(str(0) + '\t' + gold_fn)
    (goldMap, predictionMap) = bleu.computeMaps_train(pre, truth)
    bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    if(bleu_score>=threshold1):
        # words = pred[i].split()
        # test01[i]["docstring_tokens"]=words
        train01.append(test01[i])
with open('data/java/train02.jsonl', 'w') as f:
      # 遍历列表中的每个元素
      for item in train01:
          # 将元素转换为JSON格式字符串
          json_string = json.dumps(item)
          # 将JSON字符串写入文件，并在行末添加换行符
          f.write(json_string + '\n') 