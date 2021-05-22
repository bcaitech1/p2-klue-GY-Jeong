from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

from importlib import import_module
from pathlib import Path
import glob
import re
import os

def inference_soft_ensemble(args, model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=args.batch_size, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    # result = np.argmax(logits, axis=-1)
    # print(result)
    output_pred.append(logits)

  return np.concatenate(np.array(output_pred).squeeze(), axis=0)

def load_test_dataset(dataset_dir, tokenizer, mode):
  test_dataset = load_data(dataset_dir, mode)
  test_label = test_dataset['label'].values
  
  # tokenizing dataset
  if mode == 'default':
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  elif mode == 'tem':
    special_tokens = ['α', 'β', '@', '#']
    tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
    tokenized_test = tokenized_dataset_TEM(test_dataset, tokenizer)
  elif mode == 'tem_new':
    special_tokens = ['α', 'β', '@', '#']
    tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
    tokenized_test = tokenized_dataset_TEM_new(test_dataset, tokenizer)

  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  mode_list = ['default', 'default', 'tem']
  model_dir_list = ['./results/xrl-full2/checkpoint-900', './results/xrl-full-val0.1/checkpoint-810', \
    './results/xrl-full-tem/checkpoint-900']

  logits_list = []
  for mode, model_dir in tqdm(zip(mode_list, model_dir_list)):
    # load tokenizer
    MODEL_NAME = args.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load test datset
    if mode == 'default' or mode == 'tem_new':
      test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    elif mode == 'tem':
      test_dataset_dir = "/opt/ml/input/data/test/ner_test_ver2.tsv"

    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, mode)
    test_dataset = RE_Dataset(test_dataset ,test_label)

    print(len(tokenizer))

    # load my model
    model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
    model = model_module.from_pretrained(model_dir)

    if mode == 'tem':
      model.resize_token_embeddings(len(tokenizer)+4)

    model.parameters
    model.to(device)

    # predict answer
    pred_answer = inference_soft_ensemble(args, model, test_dataset, device)
    print(pred_answer.shape)
    logits_list.append(pred_answer)

  logits_answer = logits_list[0] * 0.5 + logits_list[1] * 0.3 + logits_list[2] * 0.2
  print(logits_answer)
  result = np.argmax(logits_answer, axis=-1)
  print(result)

  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame(result, columns=['pred'])
  output.to_csv(os.path.join(args.out_path, f'output.csv'), index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--out_path', type=str, default="./prediction")  
  parser.add_argument('--model_type', type=str, default='XLMRoberta')
  parser.add_argument('--pretrained_model', type=str, default='xlm-roberta-large')
  parser.add_argument('--batch_size', type=int, default=100)
  args = parser.parse_args()

  output_dir = args.out_path
  os.makedirs(output_dir, exist_ok=True)

  print(args)
  main(args)
  
