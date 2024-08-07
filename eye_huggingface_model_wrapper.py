"""
HuggingFace Model Wrapper
--------------------------
"""

import torch
import transformers

import textattack
import torch.nn.functional as F
from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, prototype, args):
        assert isinstance(
            model, transformers.PreTrainedModel
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.prototype = prototype

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            256
            if self.tokenizer.model_max_length == int(1e30)
            else 256
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)
            _, preds = outputs.logits.max(dim=-1)


        # virtual_labels = preds.to(model_device)
        model_inputs = perturb_input(self.model, inputs_dict, preds, self.prototype, self.args)
        # model_inputs = inputs_dict
        with torch.no_grad():
            outputs = self.model(**model_inputs)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            # print(outputs.logits)
            return outputs.logits

    # def get_grad(self, text_input):
    #     """Get gradient of loss with respect to input tokens.
    #
    #     Args:
    #         text_input (str): input string
    #     Returns:
    #         Dict of ids, tokens, and gradient as numpy array.
    #     """
    #     if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
    #         raise NotImplementedError(
    #             "`get_grads` for T5FotTextToText has not been implemented yet."
    #         )
    #
    #     self.model.train()
    #     embedding_layer = self.model.get_input_embeddings()
    #     original_state = embedding_layer.weight.requires_grad
    #     embedding_layer.weight.requires_grad = True
    #
    #     emb_grads = []

        # def grad_hook(module, grad_in, grad_out):
        #     emb_grads.append(grad_out[0])
        #
        # emb_hook = embedding_layer.register_backward_hook(grad_hook)
        #
        # self.model.zero_grad()
        # model_device = next(self.model.parameters()).device
        # input_dict = self.tokenizer(
        #     [text_input],
        #     add_special_tokens=True,
        #     return_tensors="pt",
        #     padding="max_length",
        #     truncation=True,
        # )
        # input_dict.to(model_device)
        # predictions = self.model(**input_dict).logits
        #
        # try:
        #     labels = predictions.argmax(dim=1)
        #     loss = self.model(**input_dict, labels=labels)[0]
        # except TypeError:
        #     raise TypeError(
        #         f"{type(self.model)} class does not take in `labels` to calculate loss. "
        #         "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
        #         "(instead of `transformers.AutoModelForSequenceClassification`)."
        #     )
        #
        # loss.backward()
        #
        # # grad w.r.t to word embeddings
        # grad = emb_grads[0][0].cpu().numpy()
        #
        # embedding_layer.weight.requires_grad = original_state
        # emb_hook.remove()
        # self.model.eval()
        #
        # output = {"ids": input_dict["input_ids"], "gradient": grad}
        #
        # return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]


def perturb_input(model, model_inputs, labels, prototype, args):

    prototype = prototype
    mean = torch.mean(prototype, dim=0)
    cfeature = prototype - mean
    cov = cfeature.t() @ cfeature / prototype.shape[0]
    inv_cov = torch.inverse(cov)

    model.eval()
    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    embedding_init = word_embedding_layer(input_ids)

    # initialize delta
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

    for astep in range(args.adv_steps):
        model.zero_grad()
        # (0) forward
        delta.requires_grad_()
        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}

        if astep == args.adv_steps - 1:
            break
        #
        # # (1) backward  这里和判断条件交换了位置，保证计算图没有被保留
        outputs = model(**batch)

        logits = outputs.logits
        losses = F.cross_entropy(logits, labels)
        loss = torch.mean(losses)
        # loss = loss / args.adv_steps
        # loss.backward()

        # md loss
        # target = outputs.pooler_output
        # target = target - mean.detach()
        # md_matrix = torch.sqrt(target @ inv_cov.detach() @ target.t())
        # mds = torch.diag(md_matrix)
        # loss += 0.5 * torch.mean(mds)
        loss = loss / args.adv_steps
        loss.backward()


        #
        # # (2) get gradient on delta
        delta_grad = delta.grad.clone().detach()
        #
        # (3) update and clip
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta - args.adv_lr * delta_grad / denorm).detach()
            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()
        elif args.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                     1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta - args.adv_lr * delta_grad / denorm).detach()

        embedding_init = word_embedding_layer(input_ids)

    return batch
