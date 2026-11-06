"""Fail on undeclared prompt truncation instead of silently losing evidence."""
from KGLMF.src.backends.huggingface import HFBackend
import torch
import time
from torch.nn import functional as F


class AlignedBackend(HFBackend):
    def __init__(self, config):
        super().__init__(config)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.resolved['training_gradient_checkpointing'] = 'non_reentrant'
        self.resolved['training_saved_tensors'] = 'cpu_pinned_offload'
        self.resolved['likelihood_logits'] = 'target_prediction_positions_only'

    def loss(self, prompt, target, prefix):
        # HF activates layer checkpointing in training mode. All weights stay frozen,
        # and dropout stays disabled so recomputation does not change the objective.
        modes = [(module,module.training) for module in self.model.modules()]
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        try:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                ids=self.prompt_ids(prompt)
                gold=self.tokenizer(target,add_special_tokens=False,return_tensors='pt',
                                    truncation=True,max_length=self.max_new).input_ids.to(self.device)
                if gold.shape[1]==0:
                    raise ValueError('Empty likelihood target')
                inputs=self.continuous_inputs(torch.cat((ids,gold),dim=1),prefix)
                # The last prompt position predicts gold[0]; exclude the last gold
                # position, which would predict a token outside the target.
                logits=self.model(**inputs,use_cache=False,logits_to_keep=gold.shape[1]+1).logits
                return target_only_nll(logits,gold)
        finally:
            for module,mode in modes:
                module.training=mode

    @torch.no_grad()
    def entropy(self,prompt):
        ids=self.prompt_ids(prompt);self.synchronize();start=time.perf_counter()
        logits=self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                          logits_to_keep=1).logits[0,-1].float()
        p=logits.softmax(-1)
        entropy=float(-(p*p.clamp_min(1e-12).log()).sum())
        self.synchronize()
        return entropy,dict(probe_tokens=ids.shape[1],probe_seconds=time.perf_counter()-start)

    def prompt_ids(self, prompt):
        if self.tokenizer.chat_template:
            options = {'enable_thinking':self.config['enable_thinking']} if 'enable_thinking' in self.config else {}
            prompt = self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                       tokenize=False,add_generation_prompt=True,**options)
            add_special = False
        else:
            add_special = True
        ids = self.tokenizer(prompt,add_special_tokens=add_special,return_tensors='pt',truncation=False).input_ids
        if ids.shape[1] > self.max_input:
            raise ValueError(f'Prompt exceeds declared input limit: {ids.shape[1]} > {self.max_input}; no silent truncation')
        return ids.to(self.device)


def target_only_nll(tail_logits, gold):
    if tail_logits.shape[1] != gold.shape[1]+1:
        raise ValueError('Expected the last prompt position and all target positions')
    return F.cross_entropy(tail_logits[:,:-1].float().reshape(-1,tail_logits.shape[-1]),gold.reshape(-1))
