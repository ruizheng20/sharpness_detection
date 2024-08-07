# coding=utf-8
"""
Attack Module
"""
import os
import sys
sys.path.append("/root/RobustRepository")

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from tqdm import tqdm
import detect.utils as utils
from torch.utils.data import DataLoader
import torch
import argparse
import csv
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
from textattack.attack_recipes import (PWWSRen2019,
                                       BAEGarg2019,
                                       TextBuggerLi2018,
                                       TextFoolerJin2019,
                                       )
from detect.modeling_bert import BertForSequenceClassificationCustomized
from detect.eye_huggingface_model_wrapper import HuggingFaceModelWrapper
# from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.constraints.pre_transformation import InputColumnModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding

logger = logging.getLogger(__name__)


def build_default_attacker(args, model) -> Attack:
    attacker = None
    if args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    else:
        print("Not implement attck!")
        exit(41)
    input_column_modification0= InputColumnModification(["sentence1", "sentence2"], {"sentence1"})
    input_column_modification1 = InputColumnModification(["sentence", "question"], {"sentence"})
    attacker.pre_transformation_constraints.append(input_column_modification0)
    attacker.pre_transformation_constraints.append(input_column_modification1)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def build_weak_attacker(args, model) -> Attack:
    attacker = None
    if args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    else:
        print("Not implement attck!")
        exit(41)

    if args.attack_method in ['bertattack']:
        attacker.transformation = WordSwapEmbedding(max_candidates=args.neighbour_vocab_size)
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
            if isinstance(constraint, UniversalSentenceEncoder):
                attacker.constraints.remove(constraint)

    # attacker.constraints.append(MaxWordsPerturbed(max_percent=args.modify_ratio))
    use_constraint = UniversalSentenceEncoder(
        threshold=args.sentence_similarity,
        metric="cosine",
        compare_against_original=True,
        window_size=15,
        skip_text_shorter_than_window=False,
    )
    attacker.constraints.append(use_constraint)
    input_column_modification0= InputColumnModification(["sentence1", "sentence2"], {"sentence1"})
    input_column_modification1 = InputColumnModification(["sentence", "question"], {"sentence"})
    attacker.pre_transformation_constraints.append(input_column_modification0)
    attacker.pre_transformation_constraints.append(input_column_modification1)
    attacker.goal_function = UntargetedClassification(model)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def attack_parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument("--model_name_or_path", default='/root/save_models/baselines/fine-tune_20220326/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs3_seed42/epoch2', type=str)
    parser.add_argument("--attack_log", default='attack_log.csv', type=str)
    parser.add_argument("--official_log", default='official_log.csv', type=str)
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default='sst2', type=str)
    parser.add_argument("--num_examples", default=1000, type=int)  # number of attack sentences
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--attack_method', type=str, default='textfooler')
    parser.add_argument("--neighbour_vocab_size", default=10, type=int)
    parser.add_argument("--modify_ratio", default=0.15, type=float)
    parser.add_argument("--sentence_similarity", default=0.85, type=float)
    parser.add_argument("--save_perturbed", default=1, type=int)
    parser.add_argument("--perturbed_file", default='results.csv', type=str)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=128)

    # Generate adversarial examples
    parser.add_argument('--c', default=0.7, type=float)

    parser.add_argument('--adv_steps', default=3, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv_init_mag', default=0.01, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_lr', default=0.01, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    # linf也行,但是需要的adv lr相应的更小,感觉性能上没有l2好
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')

    args = parser.parse_args()
    return args


def main():
    args = attack_parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassificationCustomized.from_pretrained(args.model_name_or_path, config=config)  # for finetune
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.to(device)

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

    features = get_features(model, train_loader, device)

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer, features, args)

    if args.attack_method == "bertattack":
        attack = build_weak_attacker(args, model_wrapper)
    else:
        attack = build_default_attacker(args, model_wrapper)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        attack_valid = 'test'
    else:
        attack_valid = 'validation'

    dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=attack_valid)

    # for attack
    attack_args = AttackArgs(num_examples=args.num_examples, log_to_csv=args.official_log,
                             disable_stdout=True, random_seed=args.seed)
    attacker = Attacker(attack, dataset, attack_args)
    num_results = 0
    num_successes = 0
    num_failures = 0
    for result in attacker.attack_dataset():
        logger.info(result)
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

        if args.save_perturbed:
            with open(args.perturbed_file, 'a', encoding='utf-8', newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([result.perturbed_result.attacked_text.text, result.perturbed_result.ground_truth_output])

    logger.info("[Succeeded / Failed / Total] {} / {} / {}".format(num_successes, num_failures, num_results))
    # compute metric
    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results

    if original_accuracy != 0:
        attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    else:
        attack_succ = 0
    out_csv = open(args.attack_log, 'a', encoding='utf-8', newline="")
    csv_writer = csv.writer(out_csv)
    csv_writer.writerow([args.model_name_or_path, original_accuracy, accuracy_under_attack, attack_succ])
    out_csv.close()


def get_features(model, dataset, device):

    pbar = tqdm(dataset)
    pooler_outputs = None

    model.eval()
    iter=0
    # with torch.no_grad():
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        iter += 1
        if iter > 200:
            break
        model.zero_grad()

        attack_labels = labels

        model.eval()
        model_outputs = model(**model_inputs)
        logits = model_outputs.logits

        pooler_output = model_outputs.pooler_output



        if pooler_outputs is None:
            pooler_outputs = pooler_output
        else:
            pooler_outputs = torch.cat((pooler_outputs, pooler_output.detach()), dim=0)

    return pooler_outputs



if __name__ == "__main__":
    main()
