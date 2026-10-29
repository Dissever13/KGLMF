"""One HF backend for every internal condition; no Ollama/HF confounding."""
import time
import torch
from torch.nn import functional as F


def completion_ids(sequences, input_length, used_inputs_embeds):
    # HF decoder-only generation from inputs_embeds returns generated IDs only.
    return sequences[0] if used_inputs_embeds else sequences[0, input_length:]


class HFBackend:
    def __init__(self, config):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, AutoModelForImageTextToText
        self.config = config
        self.device = config.get("device", "cuda")
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[config.get("dtype", "bfloat16")]
        common = dict(local_files_only=config.get("local_files_only", True),
                      trust_remote_code=config.get("trust_remote_code", False))
        model_kwargs = dict(common, revision=config.get("revision"))
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], **model_kwargs)
        model_config = AutoConfig.from_pretrained(config["model"], **model_kwargs)
        factory = AutoModelForImageTextToText if model_config.model_type in {"gemma4","qwen3_5"} else AutoModelForCausalLM
        self.model = factory.from_pretrained(config["model"], torch_dtype=dtype, **model_kwargs).to(self.device)
        self.model.eval().requires_grad_(False)
        self.hidden = self.model.get_input_embeddings().weight.shape[1]
        self.ple_model = next((module for module in self.model.modules()
                               if hasattr(module,"get_per_layer_inputs") and getattr(module,"hidden_size_per_layer_input",0)),None)
        enc_kwargs = dict(common, revision=config.get("encoder_revision"))
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(config["encoder"], **enc_kwargs)
        self.encoder = AutoModel.from_pretrained(config["encoder"], **enc_kwargs).to(self.device)
        self.encoder.eval().requires_grad_(False)
        self.width = self.encoder.config.hidden_size
        self.max_input = config.get("max_input_tokens", 2048)
        self.max_new = config.get("max_new_tokens", 256)
        self.temperature = config.get("temperature", 0.)
        self.resolved = {"model": config["model"], "model_commit": getattr(self.model.config,"_commit_hash",None),
                         "encoder": config["encoder"], "encoder_commit": getattr(self.encoder.config,"_commit_hash",None),
                         "transformers_version": __import__("transformers").__version__,
                         "torch_version": str(torch.__version__), "device": str(self.device),
                         "hardware": torch.cuda.get_device_name() if str(self.device).startswith("cuda") else "cpu",
                         "dtype": str(dtype), "batch_size": 1, "quantization": "none",
                         "max_input_tokens":self.max_input,"max_new_tokens":self.max_new,
                         "temperature":self.temperature,"top_p":1.,"top_k":0,"repetition_penalty":1.,
                         "input_embedding_scale":float(getattr(self.model.get_input_embeddings(),"embed_scale",1.)),
                         "truncation_side":self.tokenizer.truncation_side}

    def synchronize(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def encode(self, text):
        batch = self.encoder_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        return F.normalize((outputs*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)[0].float()

    @torch.no_grad()
    def encode_record(self, text):
        """Frozen token representations for a typed memory record (not a pooled query)."""
        batch = self.encoder_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        states = self.encoder(**batch).last_hidden_state[0]
        return states[batch["attention_mask"][0].bool()].detach().float().cpu()

    def prompt_ids(self, prompt):
        if self.tokenizer.chat_template:
            options = {"enable_thinking": self.config["enable_thinking"]} if "enable_thinking" in self.config else {}
            prompt = self.tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True, **options)
            add_special = False
        else:
            add_special = True
        return self.tokenizer(prompt, add_special_tokens=add_special, return_tensors="pt",
                              truncation=True, max_length=self.max_input).input_ids.to(self.device)

    def embeddings(self, ids, prefix):
        # Backbone parameters frozen; keep autograd through input prefix in loss().
        text = self.model.get_input_embeddings()(ids)
        if prefix is None:
            return text
        scale = getattr(self.model.get_input_embeddings(),"embed_scale",1.)
        scaled_prefix = prefix.to(device=text.device,dtype=text.dtype)*scale
        return torch.cat((scaled_prefix[None],text),dim=1)

    def continuous_inputs(self, ids, prefix):
        embeds = self.embeddings(ids,prefix)
        kwargs = {"inputs_embeds":embeds,
                  "attention_mask":torch.ones(embeds.shape[:2],dtype=torch.long,device=embeds.device)}
        if self.ple_model is not None:
            # Gemma4 needs token-identity PLE in addition to main embeddings.
            # Real text uses its exact token IDs; continuous prefix identity is zero.
            # Its context projection remains differentiable through the frozen model.
            ple = self.ple_model.get_per_layer_inputs(ids,None)
            if prefix is not None:
                zeros = ple.new_zeros((ple.shape[0],len(prefix),*ple.shape[2:]))
                ple = torch.cat((zeros,ple),dim=1)
            kwargs["per_layer_inputs"] = ple
        return kwargs

    def loss(self, prompt, target, prefix):
        ids = self.prompt_ids(prompt)
        gold = self.tokenizer(target, add_special_tokens=False, return_tensors="pt",
                              truncation=True,max_length=self.max_new).input_ids.to(self.device)
        if gold.shape[1] == 0:
            raise ValueError("Empty training target")
        sequence = torch.cat((ids,gold),dim=1)
        inputs = self.continuous_inputs(sequence,prefix)
        embeds = inputs["inputs_embeds"]
        labels = torch.full(embeds.shape[:2],-100,dtype=torch.long,device=embeds.device)
        labels[:,-gold.shape[1]:] = gold
        return self.model(**inputs,labels=labels,use_cache=False).loss

    @torch.no_grad()
    def entropy(self, prompt):
        ids = self.prompt_ids(prompt)
        self.synchronize()
        start = time.perf_counter()
        logits = self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float()
        p = logits.softmax(-1)
        entropy = float(-(p*p.clamp_min(1e-12).log()).sum())
        self.synchronize()
        return entropy, {"probe_tokens":ids.shape[1],"probe_seconds":time.perf_counter()-start}

    @torch.no_grad()
    def lexical_anchors(self, anchors):
        retained,x,y = [],[],[]
        embedding = self.model.get_input_embeddings().weight
        for word in anchors:
            ids = self.tokenizer.encode(word,add_special_tokens=False)
            if len(ids) == 1:
                retained.append(word)
                x.append(self.encode(word).cpu())
                y.append(embedding[ids[0]].detach().float().cpu())
        if not retained:
            raise ValueError("No single-token lexical anchors for this backbone")
        norm_sum = 0.
        for chunk in embedding.detach().split(4096):
            norm_sum += float(chunk.float().norm(dim=-1).sum())
        return torch.stack(x),torch.stack(y),norm_sum/len(embedding),retained

    @torch.no_grad()
    def generate(self, prompt, prefix=None):
        ids = self.prompt_ids(prompt)
        kwargs = dict(max_new_tokens=self.max_new,do_sample=self.temperature>0,
                      pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                      eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
        if self.temperature>0:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"],kwargs["top_k"] = 1.,0
        kwargs["repetition_penalty"] = 1.
        if prefix is None:
            kwargs.update(input_ids=ids,attention_mask=torch.ones_like(ids))
        else:
            kwargs.update(self.continuous_inputs(ids,prefix))
        self.synchronize()
        start = time.perf_counter()
        result = self.model.generate(**kwargs)
        self.synchronize()
        generated = completion_ids(result,ids.shape[1],prefix is not None)
        return self.tokenizer.decode(generated,skip_special_tokens=True), {
            "prompt_tokens":ids.shape[1],"output_tokens":len(generated),
            "latent_positions":0 if prefix is None else len(prefix),
            "generation_seconds":time.perf_counter()-start}
