import sys

sys.path.append("")
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


def fosc_adv_detect(args, model, target_loader, threshold, device):
    criteria = None
    attack_foscs = None
    if args.detect_type == 'fosc_sharp':  # loss在收到扰动后的变化量作为评价指标, 融入fosc进行自适应的扰动

        model.eval()
        attack_features, attack_logits, attack_losses, attack_preds, attack_labels, attack_foscs = \
            FOSC_get_features(model, target_loader, device, args, do_perturb=1)
        print('Target + Perturb Accuracy', (sum(attack_preds == attack_labels) / len(attack_labels)).item())

        model.eval()
        # target feature 没有收到扰动的
        target_features, target_logits, target_losses, target_preds, target_labels, taget_fosc = \
            FOSC_get_features(model, target_loader, device, args, do_perturb=0)
        # criteria = foscs
        print('Target Accuracy', (sum(target_preds == target_labels) / len(target_labels)).item())
        # criteria = attack_losses - target_losses
        criteria = attack_losses  # todo 1225 就改了这一句
    predict_labels = (criteria > threshold).int()

    return attack_foscs, criteria


def FOSC_get_features(model, dataset, device, args, do_perturb=0):
    pbar = tqdm(dataset)
    pooler_outputs, predict_logits, model_preds, true_labels, virtual_losses \
        = None, None, None, None, None
    foscs = None

    model.eval()
    iter = 0
    # with torch.no_grad():
    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        iter += 1
        if iter > 200:
            break
        model.zero_grad()

        attack_labels = labels

        fosc = labels  # 随便给个

        if do_perturb:
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

    return pooler_outputs, predict_logits, virtual_losses, model_preds, true_labels, foscs


def FOSC_perturb_input(model, model_inputs, labels, args):  # adv examples

    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    embedding_init = word_embedding_layer(input_ids)

    if args.adv_init_mag > 0:
        input_mask = attention_mask.to(embedding_init)
        input_lengths = torch.sum(input_mask, 1)
        if args.adv_norm_type == 'l2':
            delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = input_lengths * embedding_init.size(-1)
            magnitude = args.adv_init_mag / torch.sqrt(dims)
            delta = (delta * magnitude.view(-1, 1, 1))
        elif args.adv_norm_type == 'linf':
            delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
                                                              args.adv_init_mag) * input_mask.unsqueeze(2)
    else:  
        delta = torch.zeros_like(embedding_init)


    total_loss = 0.0
    for astep in range(args.adv_steps):
        # (0) forward
        delta.requires_grad_()
        # embedding_init.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
        logits = model(**batch).logits

        # (1) backward  这里和判断条件交换了位置，保证计算图没有被保留
        losses = F.cross_entropy(logits, labels.squeeze(-1))
        loss = torch.mean(losses)
        total_loss += loss.item()
        loss.backward()

        # (2) get gradient on delta
        delta_grad = delta.grad.clone().detach()

        # todo: fosc section
        grad_norm = torch.pow(torch.norm(delta_grad.view(delta_grad.size(0), -1).float(), p=2, dim=1), 2).detach()
        fosc = torch.ones_like(grad_norm)
        epsilon = np.sqrt(args.adv_max_norm)

        for i in range(delta_grad.shape[0]):
            fosc[i] = torch.abs(-torch.dot(delta_grad[i].view(-1), delta[i].view(-1)) + epsilon * grad_norm[i])
        if args.do_adap_size:
            # todo 做不做消融
            if astep == 0:
                adap_adv_size = AdaptiveAdvSize(fosc, args.fosc_c, args.adv_lr, args.warmup_step)

            adap_lr = adap_adv_size.update(fosc, astep).to(delta).view(-1, 1, 1)
        else:
            adap_lr = args.adv_lr

        # print("step {}, fosc {}, max len {}".format(astep, fosc.mean(), delta_grad.shape[1]))

        if astep == args.adv_steps - 1:
            embedding_init = word_embedding_layer(input_ids)
            batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            break

        # (3) update and clip
        if args.adv_norm_type == "l2":

            # adversarial noise
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            # delta = (delta + fosc_mask * args.adv_lr * delta_grad / denorm).detach()
            delta = (delta + adap_lr * delta_grad / denorm).detach()

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


class AdaptiveAdvSize:
    """
    自适应调整adv step, 由初始的c_0和最终的c_min决定的线性变化;
    """

    def __init__(self, c_0, c_min, adv_size, warmup_steps):
        self._c0 = c_0
        self._cmin = c_min
        self._alpha = adv_size
        self.warmup_steps = warmup_steps

    def update(self, ct, currect_step):

        alpha_t = (self._cmin - ct) / (self._cmin - self._c0 + 1e-12) * self._alpha
        if currect_step < self.warmup_steps:
            return torch.clamp(alpha_t, min=self._alpha)
        else:
            return torch.clamp(alpha_t, min=0)