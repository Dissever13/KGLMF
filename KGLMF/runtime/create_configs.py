"""Regenerate example configs only, never run experiments."""
from pathlib import Path
import sys
from .protocol import VERSION, write_json


def example_config(model):
    base = dict(name="kg_multi",family="latent",agents=3,k=4,budget=16,budget_policy="entropy")
    def condition(name, **overrides):
        return base | {"name":name} | overrides
    return {
        "protocol":VERSION,"seeds":[42,43,44,45,46],
        "backend":{"model":model,"revision":None,"encoder":"sentence-transformers/all-MiniLM-L6-v2",
                   "encoder_revision":None,"local_files_only":True,"trust_remote_code":False,
                   "device":"cuda","dtype":"bfloat16","max_input_tokens":2048,"max_new_tokens":512,"temperature":1.},
        "memory":{"max_tokens":32,"write_tokens":16,"temperature":.1,"hops":4,"alpha":.5,"dropout":.1,
                  "max_nodes":2000,"per_type_capacity":500,"initial_examples":500,"edge_threshold":.1,"neighbors":None,
                  "merge_threshold":.85,"prune_threshold":.1,"capsule_threshold":.3,
                  "redundancy_threshold":.95,"decay":.01,"edge_coefficients":[.2]*5,"utility_coefficients":[1/3]*3},
        "controller":{"min_tokens":12,"max_tokens":32,"rule":"floor_dev_max_entropy"},
        "training":{"epochs":3,"learning_rate":.0001,"k":4,"budget":16,"budgets":[12,16,24,32],
                    "gradient_clip":1.,"l2":.0001,"cost_weight":.001,"batch_size":1},
        "scoring":{"command":[sys.executable,"-m","KGLMF.runtime.grader","--image","kglmf-evaluator:1"],"timeout_seconds":180},
        "probe_interval":20,
        "suite_domains":{"ablation":["reasoning","math","science","code"],
                         "budget":["reasoning","math","science"],
                         "transfer":["reasoning","science"],"stream":["reasoning","qa","science"]},
        "suites":{
            "main":[condition("none_single",family="none",agents=1),condition("none_multi",family="none"),
                    condition("text_single",family="text",agents=1),condition("text_multi",family="text"),
                    condition("kg_single",agents=1),condition("kg_multi")],
            "ablation":[condition("kg_multi"),condition("flat",topology="flat"),condition("fixed_gate",gate="fixed"),
                        condition("shuffled_graph",topology="shuffled"),condition("no_role",role_condition=False),
                        condition("one_hop",hops=1),condition("kg_single",agents=1),condition("no_capsule",capsules=False)],
            "budget":[condition(f"fixed_{b}",budget=b,budget_policy="fixed") for b in [4,8,12,16,24,32]]+
                     [condition("entropy",budget_policy="entropy"),
                      condition("random_budget",budget_policy="random",budgets=list(range(12,33)))],
            "transfer":[condition(scope,scope=scope,matched_capacity=64) for scope in ["source","target","full"]],
            "stream":[condition("static",evolution="static"),condition("bounded_append",evolution="append",capsules=False),
                      condition("evolve",evolution="full"),condition("evolve_no_capsule",evolution="full",capsules=False)]}}


if __name__ == "__main__":
    from .online_protocol import example_config as online_config
    root = Path(__file__).parent/"configs"
    for name,model in [("qwen","D:/HF_models/models--Qwen--Qwen3.5-4B"),
                       ("gemma","D:/HF_models/models--google--gemma-4-E2B-it")]:
        write_json(root/(name+".json"),online_config(model))
