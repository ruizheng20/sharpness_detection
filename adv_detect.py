"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
import sys

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("..")

import torch

from torch.utils.data import DataLoader
from detect.modeling_bert import BertForSequenceClassificationCustomized
from detect.eye import detect
from detect.new_eye import fosc_adv_detect

import detect.utils as utils

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='/root/save_models/baselines/fine-tune_20220326/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs3_seed42/epoch2')
    parser.add_argument('--adv_file', type=str, default='/root/RobustRepository/detect/bert_sst2_only-attack.csv')

    parser.add_argument('--detect_type', type=str, default='fosc_sharp')
    # 现在感觉sharpness是可行的;

    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default='sst2', type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/robust_transfer/saved_models/sst-2'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='train')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # Generate adversarial examples
    parser.add_argument('--fosc_c', default=0.1, type=float)
    parser.add_argument('--warmup_step', default=4, type=int)
    parser.add_argument('--do_adap_size', default=None, type=bool)
    parser.add_argument('--adv_steps', default=4, type=int,
                        help='Number of gradient a'
                             'scent steps for the adversary')
    parser.add_argument('--adv_init_mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_lr', default=0.03, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_max_norm', default=0.2, type=float,
                        help='adv_max_norm = 0 means unlimited')

    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    # linf也行,但是需要的adv lr相应的更小,感觉性能上没有l2好
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')

    # hyper-parameters
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
    return args


def main(args):
    set_seed(args.seed)

    # pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config, tokenizer, model = load_pretrained_models(args.model_name, args.model_type, args)
    model.to(device)
    # datasets
    train_loader, dev_loader, test_loader, adv_loader = load_datasets(args, tokenizer)

    threshold = 0.25
    print('Clean Test Examples')
    test_fosc, test_criteria = fosc_adv_detect(args, model, test_loader, threshold, device)
    print('Adversarial Examples')
    adv_fosc, adv_criteria = fosc_adv_detect(args, model, adv_loader, threshold, device)

    # pic
    vars_criteria = {'adv': adv_criteria.detach().cpu().numpy(),
                     'test': test_criteria.detach().cpu().numpy()}
    plt.figure(1)
    sns.displot(vars_criteria)
    plt.xlabel("Criteria of Test and Adv")
    plt.show()

    # pic
    vars_fosc = {'adv': adv_fosc.detach().cpu().numpy(),
                 'test': test_fosc.detach().cpu().numpy()}
    plt.figure(2)
    sns.displot(vars_fosc)
    plt.xlabel("fosc of Test and Adv")
    plt.show()

    adv_fosc = adv_fosc.detach().cpu().numpy()
    test_fosc = test_fosc.detach().cpu().numpy()

    import csv
    with open('fosc_0118.csv', mode='a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['adv, index', 'fosc'])
        for i in range(len(adv_fosc)):
            csv_writer.writerow(['adv', i+1, adv_fosc[i]])


    # import csv
    # with open('fosc_0118.csv', mode='a', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(['dev', 'index', 'fosc'])
    #     for i in range(len(test_fosc)):
    #         csv_writer.writerow(['dev', i+1, test_fosc[i]])




def load_datasets(args, tokenizer):
    # dataloader
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
    split_ratio = 0.1
    train_size = round(int(len(train_dataset) * (1 - split_ratio)))
    dev_size = int(len(train_dataset)) - train_size
    # train and dev dataloader
    train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    else:
        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # for adv dataloader
    adv_dataset = utils.local_dataset(args, tokenizer, name_or_dataset=args.adv_file)
    adv_loader = DataLoader(adv_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    return train_loader, dev_loader, test_loader, adv_loader


def load_pretrained_models(model_name, model_type, args):
    if model_type == 'bert':
        config = BertConfig.from_pretrained(model_name, num_labels=args.num_labels)
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=args.do_lower_case)
        model = BertForSequenceClassificationCustomized.from_pretrained(model_name, config=config)  # for finetune
    else:
        config = AutoConfig.from_pretrained(model_name, num_labels=args.num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=args.do_lower_case)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return config, tokenizer, model


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
