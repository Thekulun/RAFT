from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu

#gold_fn = ""
#output_fn = ""
#lang = "c_sharp"
#bleu = round(_bleu(gold_fn, output_fn), 2)
#codebleu = round(calc_code_bleu.get_codebleu(gold_fn, output_fn, lang)*100, 4)

#print(bleu, codebleu)

gold_fn = "/home/fdse/sjw/ISSTA22-CodeStudy/Task/Code-Summarization/codet5/saved_models/prediction/test_best-bleu.gold"
output_fn = "/home/fdse/sjw/ISSTA22-CodeStudy/Task/Code-Summarization/codet5/saved_models/prediction/test_best-bleu.output"

bleu = _bleu(gold_fn, output_fn)

print(bleu)
