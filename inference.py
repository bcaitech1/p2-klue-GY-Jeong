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

def inference(args, model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=args.batch_size, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      # print(data)
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          # token_type_ids=data['token_type_ids'].to(device)
          # labels=data['labels'].unsqueeze(dim=-1).to(device)
          )
    # print(outputs)
    logits = outputs[0]
    # print(outputs[0])
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
    print(result)
    # exit()
    output_pred.append(result)

  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer, mode):
  test_dataset = load_data(dataset_dir, mode)
  test_label = test_dataset['label'].values
  # print(test_dataset, test_label)
  # tokenizing dataset

  if args.mode == 'default':
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  elif args.mode == 'tem':
    special_tokens = ['α', 'β', '@', '#']
    print(len(tokenizer))
    tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
    print(len(tokenizer))
    # exit()
    tokenized_test = tokenized_dataset_TEM(test_dataset, tokenizer)
  elif args.mode == 'tem_new':
    special_tokens = ['α', 'β', '@', '#']
    print(len(tokenizer))
    tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
    print(len(tokenizer))
    # exit()
    tokenized_test = tokenized_dataset_TEM_new(test_dataset, tokenizer)


  # print(tokenized_test)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  # MODEL_NAME = args.pretrained_model
  # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  MODEL_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  print(len(tokenizer))

  # load test datset
  if args.mode == 'default' or args.mode == 'tem_new':
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  elif args.mode == 'tem':
    test_dataset_dir = "/opt/ml/input/data/test/ner_test_ver2.tsv"

  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, args.mode)
  # print(test_dataset[0])
  test_dataset = RE_Dataset(test_dataset ,test_label)
  # print(test_dataset[0])

  print(len(tokenizer))

  # load my model
  model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
  model = model_module.from_pretrained(args.model_dir)
  model.resize_token_embeddings(len(tokenizer))
  print(model.num_labels)
  # model = BertForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

# ====================
#   train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
#   #dev_dataset = load_data("./dataset/train/dev.tsv")
#   train_label = train_dataset['label'].values
#   #dev_label = dev_dataset['label'].values
  
#   # tokenizing dataset
#   tokenized_train = tokenized_dataset(train_dataset, tokenizer)
#   #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
#   print(tokenized_train[0])

#   # make dataset for pytorch.
#   RE_train_dataset = RE_Dataset(tokenized_train, train_label)
#   #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
#   print(RE_train_dataset[0])
# =======================


  # predict answer
  pred_answer = inference(args, model, test_dataset, device)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv(os.path.join(args.out_path, f'output.csv'), index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--out_path', type=str, default="./prediction")  
  parser.add_argument('--model_type', type=str, default='XLMRoberta')
  parser.add_argument('--pretrained_model', type=str, default='xlm-roberta-large')
  parser.add_argument('--model_dir', type=str, default="./results/checkpoint-500")
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--mode', type=str, default='default')
  args = parser.parse_args()

  output_dir = args.out_path
  os.makedirs(output_dir, exist_ok=True)

  print(args)
  main(args)
  
