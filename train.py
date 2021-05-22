import pickle as pickle
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,\
   RobertaTokenizer, RobertaForSequenceClassification, XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from load_data import *
from tools import *
from sklearn.model_selection import train_test_split

import argparse
from importlib import import_module

def train(args):
  # load model and tokenizer
  MODEL_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  if args.training_data_type == 'default' or args.training_data_type == 'tem_new':
    train_dataset = load_data("/opt/ml/input/data/train/train.tsv", args.mode)
  elif args.training_data_type == 'big':
    train_dataset = load_data("/opt/ml/input/data/train/train+all.tsv", args.mode)
  elif args.training_data_type == 'tem':
    train_dataset = load_data("/opt/ml/input/data/train/ner_train_ver2.tsv", args.mode)

  train_label = train_dataset['label'].values
  print(train_label, len(train_label))

  if args.validation_ratio != 0.0:
    train_dataset, dev_dataset, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=args.validation_ratio,\
                                                                          random_state = args.seed, stratify=train_label)
    print('Validation!', len(train_dataset), len(dev_dataset))
  
  print('No validation!', len(train_dataset))

  # tokenizing dataset
  if args.mode == 'default':
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    if args.validation_ratio != 0.0:
      tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
  elif args.mode == 'tem':
    special_tokens = ['α', 'β', '@', '#']
    tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
    if args.validation_ratio != 0.0:
      tokenized_dev = tokenized_dataset_TEM(dev_dataset, tokenizer)
    tokenized_train = tokenized_dataset_TEM(train_dataset, tokenizer)
  elif args.mode == 'tem_new':
    special_tokens = ['α', 'β', '@', '#']
    tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
    if args.validation_ratio != 0.0:
      tokenized_dev = tokenized_dataset_TEM_new(dev_dataset, tokenizer)
    tokenized_train = tokenized_dataset_TEM_new(train_dataset, tokenizer)
    # print(tokenizer.token)

  print(tokenized_train)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  if args.validation_ratio != 0.0:
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  config_module = getattr(import_module("transformers"), args.model_type + "Config")
  model_config = config_module.from_pretrained(MODEL_NAME)
  model_config.num_labels = 42

  model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
  model = model_module.from_pretrained(MODEL_NAME, config=model_config)
  model.resize_token_embeddings(len(tokenizer)+4)
  model.parameters
  model.to(device)

  output_dir = increment_path(args.output_dir)

  
  if args.validation_ratio != 0.0:
    evaluation_strategy = 'epoch'
  else:
    evaluation_strategy = 'no'

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    # save_steps=args.save_steps,                 # model saving step.
    save_strategy='epoch',
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    # warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    # weight_decay=args.weight_decay,               # strength of weight decay
    # logging_dir=args.logging_dir,            # directory for storing logs
    # logging_steps=args.logging_steps,              # log saving step.
    evaluation_strategy=evaluation_strategy, # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    # eval_steps = 500,            # evaluation step.
    # dataloader_num_workers = 4,
    label_smoothing_factor = args.label_smoothing_factor
  )

  if args.validation_ratio != 0.0:
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=RE_train_dataset,         # training dataset
      eval_dataset=RE_dev_dataset,             # evaluation dataset
      compute_metrics=compute_metrics         # define metrics function
    )
  
  else:
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=RE_train_dataset,         # training dataset
      # eval_dataset=RE_dev_dataset,             # evaluation dataset
      # compute_metrics=compute_metrics         # define metrics function
    )

  # train model
  trainer.train()
  # trainer.save_model(output_dir)
  # trainer.save_state()

def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
  parser.add_argument('--model_type', type=str, default='XLMRoberta')
  parser.add_argument('--pretrained_model', type=str, default='xlm-roberta-large')
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=5e-5)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--warmup_steps', type=int, default=300)               # number of warmup steps for learning rate scheduler
  parser.add_argument('--output_dir', type=str, default='./results')
  parser.add_argument('--save_steps', type=int, default=500)
  parser.add_argument('--save_total_limit', type=int, default=3)
  parser.add_argument('--logging_steps', type=int, default=100)
  parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs
  parser.add_argument('--mode', type=str, default='default')
  parser.add_argument('--validation_ratio', type=float, default=0.2)
  parser.add_argument('--label_smoothing_factor', type=float, default=0.5)
  parser.add_argument('--training_data_type', type=str, default='default')

  args = parser.parse_args()
  seed_everything(args.seed)
  
  main(args)
