"""Data provenance, split isolation, strict scoring and versioned result records.

No model imports: prepare/check/aggregate can run on a CPU without downloads.
"""
import hashlib
import json
import math
import random
import re
import string
import unicodedata
from pathlib import Path

VERSION = "kbs-v2"
DATASETS = {
    "strategyqa": ("strategyqa_real.json", "reasoning", "boolean"),
    "triviaqa": ("trivia_qa_real.json", "qa", "em"),
    "popqa": ("akariasai_PopQA_real.json", "qa", "em"),
    "hotpotqa": ("HotpotQA_real.json", "reasoning", "em_f1"),
    "bigcodebench": ("bigcodebench_real.json", "code", "external_code"),
    "kodcode": ("kodcode_real.json", "code", "external_code"),
    "math500": ("math500_real.json", "math", "external_math"),
    "mmlu_pro": ("mmlu_pro_real.json", "science", "choice"),
    "gpqa": ("gpqa_real.json", "science", "choice"),
}


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def question_key(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def input_key(question, raw, domain):
    choices = raw.get("choices",raw.get("options"))
    task = raw.get("task_id") if domain == "code" else None
    return digest([question_key(question),choices,task])


def grouping_aliases(dataset, raw, question, domain):
    identifiers = ["q:"+question_key(question)]
    native = raw.get("task_id",raw.get("question_id",raw.get("id")))
    if native is not None:
        identifiers.append(f"task:{dataset}:{native}")
        if domain == "code" and "/" in str(native):
            identifiers.append("code-task:"+str(native))
    for field in ("group_id","task_family_id","derivative_of","parent_id"):
        value = raw.get(field)
        if value is not None:
            for item in value if isinstance(value,list) else [value]:
                identifiers.append(f"task:{dataset}:{item}")
    identifiers += ["global:"+str(v) for v in raw.get("related_source_ids",[])]
    return identifiers


def first_present(item, keys):
    for key in keys:
        if key in item and item[key] is not None and item[key] != "":
            return item[key]
    raise ValueError(f"Missing one of {keys}; keys={list(item)}")


def normalize_answer(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def choice_gold(question, aliases, raw):
    choices = raw.get("choices",raw.get("options"))
    if isinstance(choices,dict) and "text" in choices:
        mapping = dict(zip(choices.get("label",list("ABCDEFGHIJ")),choices["text"]))
    elif isinstance(choices,list):
        mapping = {chr(65+i):v for i,v in enumerate(choices)}
    else:
        option_text = question.split("Options:",1)[-1]
        parts = re.split(r"\(([A-J])\)\s*",option_text)
        mapping = dict(zip(parts[1::2],parts[2::2]))
    labels = []
    for answer in aliases:
        if re.fullmatch(r"[A-J]",answer.strip(),re.I):
            if answer.strip().upper() not in mapping:
                raise ValueError("Gold option label is absent from the original choices")
            labels.append(answer.strip().upper())
        else:
            matched = [k for k,v in mapping.items() if question_key(v)==question_key(answer)]
            if len(matched) != 1:
                raise ValueError("Multiple-choice gold cannot be unambiguously mapped to an option")
            labels.extend(matched)
    return list(dict.fromkeys(labels))


def prepare_manifest(data_dir, output, datasets=None, seed=42, train_fraction=.6, dev_fraction=.2,
                     limits=None):
    """Local benchmark subsets, not official benchmark splits. Preserve raw fields.

    Global normalized question deduplication precedes all split assignment.
    Raw official split names, if present, are retained but never silently relabelled
    as official evaluation: the resulting protocol is explicitly a local resplit.
    """
    if not (0 < train_fraction < 1 and 0 < dev_fraction < 1 - train_fraction):
        raise ValueError("Require positive train/dev/test fractions")
    seen, samples, duplicates, files, rows, duplicate_aliases = {}, [], [], {}, [], {}
    for name in sorted(datasets or DATASETS):
        filename, domain, metric = DATASETS[name]
        path = Path(data_dir) / filename
        raw = read_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a JSON array")
        files[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for index, item in enumerate(raw):
            q = str(first_present(item, ["question", "prompt", "input"]))
            answer = first_present(item, ["answer", "target", "output", "answer_key"])
            if isinstance(answer, dict):
                aliases = answer.get("aliases", answer.get("value", answer.get("text")))
                if aliases is None:
                    raise ValueError(f"{name}:{index}: unsupported answer object")
            else:
                aliases = answer if isinstance(answer, list) else [answer]
            if not isinstance(aliases, list):
                aliases = [aliases]
            aliases = [str(x) for x in aliases]
            extra = item.get("aliases", [])
            aliases += [str(x) for x in extra] if isinstance(extra, list) else [str(extra)]
            if metric == "choice":
                try:
                    aliases = choice_gold(q,aliases,item)
                except ValueError as exc:
                    raise ValueError(f"{name}:{index}: {exc}. Recover original options/answer labels; do not guess or drop hard examples.") from exc
            key = input_key(q,item,domain)
            if not question_key(q):
                raise ValueError(f"Empty question at {name}:{index}")
            if key in seen:
                duplicates.append({"dataset": name, "index": index, "kept_id": seen[key]})
                duplicate_aliases.setdefault(seen[key],[]).extend(grouping_aliases(name,item,q,domain))
                continue
            sid = digest(key)[:24]
            seen[key] = sid
            rows.append({"id": sid, "dataset": name, "domain": domain, "metric": metric,
                         "question": q, "answers": list(dict.fromkeys(aliases)),
                         "source_index": index, "source_id": item.get("id", item.get("task_id")),
                         "raw": item,
                         "source_revision": item.get("source_revision"),
                         "source_quality": "requires_source_recovery" if domain == "code" and not (item.get("task_id") and item.get("test")) else "source_export"})
    # Union all known task/derivative identifiers before partitioning. Namespaced
    # native IDs avoid accidental collisions between unrelated dataset indices.
    parent = {r["id"]:r["id"] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    aliases = {}
    for row in rows:
        identifiers = grouping_aliases(row["dataset"],row["raw"],row["question"],row["domain"])+duplicate_aliases.get(row["id"],[])
        for identifier in identifiers:
            if identifier in aliases:
                parent[find(row["id"])] = find(aliases[identifier])
            else:
                aliases[identifier] = row["id"]
    groups = {}
    for row in rows:
        groups.setdefault(find(row["id"]),[]).append(row)
    from collections import Counter
    totals = Counter(r["dataset"] for r in rows)
    counts = {s:Counter() for s in ("train","dev","test")}
    fractions = dict(train=train_fraction,dev=dev_fraction,test=1-train_fraction-dev_fraction)
    for members in sorted(groups.values(),key=lambda g:digest([seed,sorted(r["id"] for r in g)])):
        amounts = Counter(r["dataset"] for r in members)
        # Greedy weighted deficit: deterministic target ratios, indivisible groups.
        def error_delta(split):
            return sum((counts[split][d]+n-fractions[split]*totals[d])**2
                       -(counts[split][d]-fractions[split]*totals[d])**2 for d,n in amounts.items())
        split = min(fractions,key=error_delta)
        gid = "group:"+digest(sorted(r["id"] for r in members))[:24]
        for row in members:
            row.update(split=split,group_id=gid)
        counts[split].update(amounts)
    # Optional caps select complete groups only; default uses all source records.
    retained_counts = {s:Counter() for s in fractions}
    for members in sorted(groups.values(),key=lambda g:digest([seed,"sample",g[0]["group_id"]])):
        split = members[0]["split"]
        amounts = Counter(r["dataset"] for r in members)
        if limits is not None and any(retained_counts[split][d]+n > limits[list(fractions).index(split)] for d,n in amounts.items()):
            continue
        samples.extend(members)
        retained_counts[split].update(amounts)
    for name in totals:
        if any(retained_counts[s][name] == 0 for s in fractions):
            raise ValueError(f"{name}: insufficient independent groups for three nonempty splits")
    manifest = {"version": VERSION, "protocol": "deduplicated_local_resplit",
                "seed": seed, "limits": list(limits) if limits else None, "fractions": list(fractions.values()),
                "group_assignment": "global_union_then_squared_deficit",
                "group_counts": {s:len({r['group_id'] for r in samples if r['split']==s}) for s in fractions},
                "source_files": files, "duplicates_removed": duplicates, "samples": samples}
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    write_json(output, manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get("version") != VERSION:
        raise ValueError("Unsupported manifest version")
    if digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest.get("sha256"):
        raise ValueError("Manifest hash mismatch")
    ids, keys, groups = set(), set(), {}
    for row in manifest["samples"]:
        key = input_key(row["question"],row["raw"],row["domain"])
        if row["id"] in ids or key in keys:
            raise ValueError("Duplicate example/question across splits")
        ids.add(row["id"])
        keys.add(key)
        gid = row.get("group_id")
        if not gid or gid in groups and groups[gid] != row["split"]:
            raise ValueError("Missing group or derivative group crosses splits")
        groups[gid] = row["split"]
        if row["split"] not in {"train", "dev", "test"} or not row["answers"]:
            raise ValueError("Invalid split or empty answers")
    return manifest


def provenance_ids(row):
    return [row["id"],row["group_id"]] if row.get("group_id") else [row["id"]]


def validate_sources(manifest):
    """Fail before model loading when recovered source fields are incomplete."""
    for row in manifest["samples"]:
        if not row.get("source_revision"):
            raise ValueError(f"{row['dataset']}: source_revision is required; recover original exports first")
        raw = row["raw"]
        if row["domain"] == "code" and not (raw.get("task_id") and raw.get("test")):
            raise ValueError(f"{row['id']}: original code task ID and executable tests required")
        if row["metric"] == "choice" and not raw.get("choices",raw.get("options")):
            raise ValueError(f"{row['id']}: preserve structured original answer options")
    return manifest


def validate_config(config):
    if config.get("protocol") != VERSION:
        raise ValueError("Use kbs-v2 configuration; old configs implement a different method")
    memory,training = config["memory"],config["training"]
    if min(memory["max_tokens"],memory["write_tokens"],memory["per_type_capacity"],memory["max_nodes"]) < 1:
        raise ValueError("Memory dimensions/capacities must be positive")
    if memory["temperature"] <= 0 or not 0 < memory["alpha"] < 1 or memory["hops"] < 0:
        raise ValueError("Invalid routing controller")
    if training["epochs"] < 1 or training["learning_rate"] <= 0 or training["gradient_clip"] <= 0:
        raise ValueError("Invalid training schedule")
    if training["l2"] < 0 or training["cost_weight"] < 0:
        raise ValueError("Cost/regularization weights must be nonnegative")
    if not training["budgets"] or min(training["budgets"]) < 1 or max(training["budgets"]) > memory["max_tokens"]:
        raise ValueError("Training budgets exceed composition query capacity")
    for suite in config["suites"].values():
        if len({c["name"] for c in suite}) != len(suite):
            raise ValueError("Duplicate condition names")
        for c in suite:
            if c.get("agents",3) not in {1,3} or c.get("family","latent") not in {"none","text","latent"}:
                raise ValueError("Unsupported agent condition")
            if c.get("k",4) < 1 or max(c.get("budgets",[c.get("budget",16)])) > memory["max_tokens"]:
                raise ValueError("Invalid condition retrieval/composition budget")
    return config


def final_answer(text):
    matches = re.findall(r"FINAL\s*:\s*(.*)", text, flags=re.I)
    return matches[-1].strip() if matches else text.strip()


def score_answer(text, row):
    """Never pretend compilation or substring overlap is code/math accuracy.

    Official code/math harness scores must be imported with generation hashes.
    The local QA metrics are stated explicitly, not claimed to be official runs.
    """
    metric, pred = row["metric"], final_answer(text)
    golds = row["answers"]
    if metric.startswith("external_"):
        return {"score": None, "status": "requires_official_harness", "prediction": pred}
    if metric == "boolean":
        def boolean(x):
            x = x.strip().lower().rstrip(".! ")
            return {"true": "yes", "1": "yes", "yes": "yes", "false": "no", "0": "no", "no": "no"}.get(x)
        p = boolean(pred)
        result = float(p is not None and p in [boolean(g) for g in golds])
    elif metric == "choice":
        def choice(x):
            m = re.fullmatch(r"\s*\(?([A-J])\)?[. ]*", x, flags=re.I)
            return m.group(1).upper() if m else None
        p = choice(pred)
        if not any(choice(g) for g in golds):
            raise ValueError(f"{row['id']}: choice gold is not a letter; normalize dataset explicitly")
        result = float(p is not None and p in [choice(g) for g in golds])
    else:
        result = float(normalize_answer(pred) in [normalize_answer(g) for g in golds])
    out = {"score": result, "status": "scored", "prediction": pred}
    if metric == "em_f1":
        from collections import Counter
        p = normalize_answer(pred).split()
        f1s = []
        for gold in golds:
            g = normalize_answer(gold).split()
            overlap = sum((Counter(p) & Counter(g)).values())
            f1s.append(2 * overlap / (len(p) + len(g)) if p or g else 1.)
        out["f1"] = max(f1s)
    return out


def bootstrap_mean(values, seed=1729, repeats=2000):
    if not values:
        raise ValueError("Empty observations")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repeats))
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v-mean)**2 for v in values) / (len(values)-1)) if len(values)>1 else None
    return {"mean": mean, "std": std, "ci95": [means[int(.025*repeats)], means[int(.975*repeats)]],
            "n": len(values), "resampling_unit": "training_seed"}
