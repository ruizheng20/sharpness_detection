import sys
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("..")
import torch
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA

from tqdm import tqdm


def detect(args, model, train_loader, target_loader, threshold, device):
    # args 里需要提供攻击参数，包括adv_steps, adv_lr;
    # train_dataset 用来构建原型统计量, model_inputs为当前需要检测的数据;

    # adv + delta feature
    model.eval()
    attack_features, attack_logits, attack_losses, attack_preds, attack_labels = \
        get_features(model, target_loader, device, args, do_perturb=1)
    print('Target + Perturb Accuracy', (sum(attack_preds == attack_labels) / len(attack_labels)).item())

    # # draw mean feature
    # mean_feature = torch.mean(attack_features, dim=0).detach().cpu().numpy()
    # plt.figure(1)
    # x = [i for i in range(len(mean_feature))]
    # plt.scatter(x, mean_feature)
    # plt.xlabel("target+perturb: mean_feature")
    # plt.show()

    criteria = None
    if args.detect_type == 'md':  # 马氏距离判别
        # 原型
        features, _, _, _, _ = get_features(model, train_loader, device, args)
        # target feature
        target_features, target_logits, target_losses, target_preds, target_labels = \
            get_features(model, target_loader, device, args)
        print('Target Accuracy', (sum(target_preds == target_labels) / len(target_labels)).item())

        # # draw mean feature
        # mean_feature = torch.mean(target_features, dim=0).detach().cpu().numpy()
        # plt.figure(1)
        # x = [i for i in range(len(mean_feature))]
        # plt.scatter(x, mean_feature)
        # plt.xlabel("target: mean_feature")
        # plt.show()

        # target_features = target_features.clamp(min=-args.c, max=args.c)
        # attack_features = attack_features.clamp(min=-args.c, max=args.c)

        target_mds = mahalanobis_distance(target=target_features, prototype=features)
        attack_mds = mahalanobis_distance(target=attack_features, prototype=features)
        criteria = attack_mds - target_mds

    elif args.detect_type == 'logit':  # logits
        # target feature
        target_features, target_logits, target_losses, target_preds, target_labels = \
            get_features(model, target_loader, device, args)
        print('Target Accuracy', (sum(target_preds == target_labels) / len(target_labels)).item())

        criteria = compare_logits(target_logits, attack_logits)

    elif args.detect_type == 'loss':  # 虚拟loss, 标签为当前预测值;

        # w = model.classifier.weight
        # b = model.classifier.bias
        #
        # attack_features = attack_features.clamp(min=-args.c, max=args.c)
        # react_logits = attack_features.matmul(w.t()) + b
        # #
        # react_attack_losses = F.cross_entropy(react_logits, attack_preds.squeeze(-1), reduction='none')
        # criteria = react_attack_losses

        criteria = attack_losses

    elif args.detect_type == 'sharpness': # loss在收到扰动后的变化量作为评价指标
        model.eval()
        # target feature 没有收到扰动的
        target_features, target_logits, target_losses, target_preds, target_labels = \
            get_features(model, target_loader, device, args, do_perturb=0)
        criteria = attack_losses - target_losses

    elif args.detect_type == 'fosc_sharp':  # loss在收到扰动后的变化量作为评价指标, 融入fosc进行自适应的扰动

        attack_features, attack_logits, attack_losses, attack_preds, attack_labels, foscs = \
            FOSC_get_features(model, target_loader, device, args, do_perturb=1)

        model.eval()
        # target feature 没有收到扰动的
        target_features, target_logits, target_losses, target_preds, target_labels = \
            get_features(model, target_loader, device, args, do_perturb=0)
        # criteria = foscs
        criteria = attack_losses - target_losses

    elif args.detect_type == 'baseline_loss':  # 虚拟loss, 标签为当前预测值;

        # target feature
        target_features, target_logits, target_losses, target_preds, target_labels = \
            get_features(model, target_loader, device, args)

        criteria = target_losses

    elif args.detect_type == 'baseline_md':
        # 原型
        features, _, _, _, _ = get_features(model, train_loader, device, args)
        # target feature
        target_features, target_logits, target_losses, target_preds, target_labels = \
            get_features(model, target_loader, device, args)

        target_mds = mahalanobis_distance(target=target_features, prototype=features)

        criteria = target_mds

    #
    predict_labels = (criteria > threshold).int()

    return predict_labels, criteria


def compare_logits(target_logits, attack_logits):

    return torch.norm(target_logits-attack_logits, p=2, dim=1)


def get_features(model, dataset, device, args, do_perturb=0):

    pbar = tqdm(dataset)
    pooler_outputs, predict_logits, model_preds, true_labels, virtual_losses\
        = None, None, None, None, None

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
        if do_perturb:
            model_outputs = model(**model_inputs)
            logits = model_outputs.logits
            _, preds = logits.max(dim=-1)
            attack_labels = preds.to(device)

            model_inputs = perturb_input(model, model_inputs, attack_labels, args)

        model.eval()
        model_outputs = model(**model_inputs)
        logits = model_outputs.logits
        _, preds = logits.max(dim=-1)

        pooler_output = model_outputs.pooler_output

        virtual_loss = F.cross_entropy(logits, attack_labels.squeeze(-1), reduction='none')
        loss = F.cross_entropy(logits, attack_labels.squeeze(-1))  # 释放计算图

        loss.backward()

        if pooler_outputs is None:
            pooler_outputs = pooler_output
            predict_logits = torch.softmax(logits, dim=1)
            virtual_losses = virtual_loss
            model_preds = preds
            true_labels = labels
        else:
            pooler_outputs = torch.cat((pooler_outputs, pooler_output), dim=0)
            predict_logits = torch.cat((predict_logits, torch.softmax(logits, dim=1)), dim=0)
            virtual_losses = torch.cat((virtual_losses, virtual_loss), dim=0)
            model_preds = torch.cat((model_preds, preds), dim=0)
            true_labels = torch.cat((true_labels, labels), dim=0)
        # if model_preds is None:
        #     model_preds = preds.detach().cpu().numpy()
        #     true_labels = labels.detach().cpu().numpy()
        # else:
        #     _, preds = logits.max(dim=-1)
        #     model_preds = np.append(model_preds, preds.detach().cpu().numpy(), axis=0)
        #     true_labels = np.append(true_labels, labels.detach().cpu().numpy(), axis=0)

    return pooler_outputs, predict_logits, virtual_losses, model_preds, true_labels


def FOSC_get_features(model, dataset, device, args, do_perturb=0):

    pbar = tqdm(dataset)
    pooler_outputs, predict_logits, model_preds, true_labels, virtual_losses\
        = None, None, None, None, None
    foscs = None

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

        model_outputs = model(**model_inputs)
        logits = model_outputs.logits
        _, preds = logits.max(dim=-1)
        attack_labels = preds.to(device)

        model_inputs, fosc = FOSC_perturb_input(model, model_inputs, attack_labels, args)

        model.eval()
        model_outputs = model(**model_inputs)
        logits = model_outputs.logits
        _, preds = logits.max(dim=-1)

        pooler_output = model_outputs.pooler_output

        virtual_loss = F.cross_entropy(logits, attack_labels.squeeze(-1), reduction='none')
        loss = F.cross_entropy(logits, attack_labels.squeeze(-1))  # 释放计算图

        loss.backward()

        if pooler_outputs is None:
            pooler_outputs = pooler_output
            predict_logits = torch.softmax(logits, dim=1)
            virtual_losses = virtual_loss
            model_preds = preds
            true_labels = labels
            foscs = fosc
        else:
            pooler_outputs = torch.cat((pooler_outputs, pooler_output), dim=0)
            predict_logits = torch.cat((predict_logits, torch.softmax(logits, dim=1)), dim=0)
            virtual_losses = torch.cat((virtual_losses, virtual_loss), dim=0)
            model_preds = torch.cat((model_preds, preds), dim=0)
            true_labels = torch.cat((true_labels, labels), dim=0)
            foscs = torch.cat((foscs, fosc), dim=0)

        # if model_preds is None:
        #     model_preds = preds.detach().cpu().numpy()
        #     true_labels = labels.detach().cpu().numpy()
        # else:
        #     _, preds = logits.max(dim=-1)
        #     model_preds = np.append(model_preds, preds.detach().cpu().numpy(), axis=0)
        #     true_labels = np.append(true_labels, labels.detach().cpu().numpy(), axis=0)

    return pooler_outputs, predict_logits, virtual_losses, model_preds, true_labels, foscs


def FOSC_perturb_input(model, model_inputs, labels, args):  # adv examples

    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    embedding_init = word_embedding_layer(input_ids)

    delta = torch.zeros_like(embedding_init)

    total_loss = 0.0
    for astep in range(args.adv_steps):
        # (0) forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch).logits

        # (1) backward  这里和判断条件交换了位置，保证计算图没有被保留
        losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / args.adv_steps
        total_loss += loss.item()
        loss.backward()

        # (2) get gradient on delta
        delta_grad = delta.grad.clone().detach()

        if astep == args.adv_steps - 1:

            # epsilon = args.adv_steps * args.adv_lr

            epsilon = args.adv_max_norm
            grad_norm = torch.norm(delta_grad.view(delta_grad.size(0), -1).float(), p=1, dim=1).detach()
            fosc = grad_norm

            for i in range(delta_grad.shape[0]):
                fosc[i] = -torch.dot(delta_grad[i].view(-1), delta[i].view(-1)) + epsilon * grad_norm[i]

            embedding_init = word_embedding_layer(input_ids)
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}

            break

        # (3) update and clip
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()
        elif args.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                     1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()

        embedding_init = word_embedding_layer(input_ids)

    return batch, fosc


def perturb_input(model, model_inputs, labels, args):  # adv examples

    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    embedding_init = word_embedding_layer(input_ids)

    delta = torch.zeros_like(embedding_init)

    # # initialize delta
    # if args.adv_init_mag > 0:
    #     input_mask = attention_mask.to(embedding_init)
    #     input_lengths = torch.sum(input_mask, 1)
    #     if args.adv_norm_type == 'l2':
    #         delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
    #         dims = input_lengths * embedding_init.size(-1)
    #         magnitude = args.adv_init_mag / torch.sqrt(dims)
    #         delta = (delta * magnitude.view(-1, 1, 1))
    #     elif args.adv_norm_type == 'linf':
    #         delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
    #                                                           args.adv_init_mag) * input_mask.unsqueeze(2)
    # else:
    #     delta = torch.zeros_like(embedding_init)

    total_loss = 0.0
    for astep in range(args.adv_steps):
        # (0) forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch).logits

        if astep == args.adv_steps - 1:
            break

        # (1) backward  这里和判断条件交换了位置，保证计算图没有被保留
        losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        loss = loss / args.adv_steps
        total_loss += loss.item()
        loss.backward()

        # (2) get gradient on delta
        delta_grad = delta.grad.clone().detach()

        # (3) update and clip
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()
        elif args.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                     1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()

        embedding_init = word_embedding_layer(input_ids)

    return batch


def mahalanobis_distance(target, prototype):

    mean = torch.mean(prototype, dim=0)
    cfeature = prototype - mean
    cov = cfeature.t() @ cfeature / prototype.shape[0]
    target = target - mean
    md_matrix = torch.sqrt(target @ torch.inverse(cov) @ target.t())
    mds = torch.diag(md_matrix)  # 对角线部分才是马氏距离,非对角线位置可以看作相关性;

    return mds