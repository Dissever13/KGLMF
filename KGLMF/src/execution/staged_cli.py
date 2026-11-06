import argparse
import json
from collections import Counter
from pathlib import Path

from KGLMF.src.data.protocol import VERSION, prepare_manifest, read_json, validate_manifest, validate_sources, validate_config, write_json


def main():
    parser = argparse.ArgumentParser(description="KG-LMF KBS v2. Explicit model commands only.")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--datasets",nargs="+")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--limits",nargs=3,type=int,help="Optional train/dev/test caps, complete groups only")
    p = sub.add_parser("check")
    p.add_argument("--config",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output")
    p = sub.add_parser("calibrate")
    p.add_argument("--config",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    for cmd in ("initialize","feedback","train","run"):
        p = sub.add_parser(cmd)
        p.add_argument("--config",required=True)
        p.add_argument("--manifest",required=True)
        p.add_argument("--output",required=True)
        p.add_argument("--seed",type=int,required=True)
        if cmd == "train":
            p.add_argument("--freeze-adapter",action="store_true")
            p.add_argument("--initial",required=True)
            p.add_argument("--feedback",required=True)
        if cmd == "feedback":
            p.add_argument("--checkpoint",required=True)
            p.add_argument("--adapter",choices=["trained","cma"])
        if cmd == "run":
            p.add_argument("--checkpoint",required=True)
            p.add_argument("--suite",choices=["main","ablation","budget","transfer","stream"],required=True)
            p.add_argument("--conditions",nargs="+")
    p = sub.add_parser("score-feedback")
    for name in ("config","manifest","feedback","output"):
        p.add_argument("--"+name,required=True)
    p = sub.add_parser("score-run")
    for name in ("config","manifest","predictions","output"):
        p.add_argument("--"+name,required=True)
    p = sub.add_parser("export-memory")
    for name in ("checkpoint","manifest","output"):
        p.add_argument("--"+name,required=True)
    p = sub.add_parser("finalize")
    for name in ("config","manifest","checkpoint","feedback","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--seed",type=int,required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--reference",default="kg_multi")
    p = sub.add_parser("failures")
    p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--output",required=True)
    p = sub.add_parser("export-harness")
    p.add_argument("--predictions",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p = sub.add_parser("import-scores")
    p.add_argument("--predictions",required=True)
    p.add_argument("--scores",required=True,nargs="+")
    p.add_argument("--output",required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if Path(args.output).exists():
            parser.error("Manifest exists; choose a versioned new output")
        m = prepare_manifest(args.data_dir,args.output,args.datasets,args.seed,limits=args.limits)
        print(json.dumps({"manifest_sha256":m["sha256"],"samples":len(m["samples"]),"duplicates":len(m["duplicates_removed"])}))
    elif args.command == "check":
        m,c = validate_sources(validate_manifest(read_json(args.manifest))),validate_config(read_json(args.config))
        if c.get("protocol") != VERSION:
            parser.error("Use kbs-v2 configuration; v1 is a different method")
        conditions = [v for suite in c["suites"].values() for v in suite]
        for v in conditions:
            if max(v.get("budgets",[v.get("budget",16)])) > c["memory"]["max_tokens"]:
                parser.error("Budget exceeds model query capacity")
        result = {"status":"configuration_checked_no_models_loaded", "counts":dict(Counter(r["dataset"]+":"+r["split"] for r in m["samples"])),
                  "manifest_sha256":m["sha256"],"model":c["backend"]["model"],
                  "seeds":c["seeds"],"suites":{k:[v["name"] for v in values] for k,values in c["suites"].items()},
                  "pending_requirements":["Set exact local model paths or pinned Hub revisions", "Configure isolated code/math scorer for training feedback and development", "Dev-only entropy calibration", "Initialize and score complete paired training feedback before train"]}
        if args.output:
            write_json(args.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command == "calibrate":
        from KGLMF.src.execution.runner import calibrate
        print(calibrate(args.config,args.manifest,args.output))
    elif args.command == "train":
        from KGLMF.src.execution.runner import train
        print(train(args.config,args.manifest,args.output,args.seed,args.freeze_adapter,args.initial,args.feedback))
    elif args.command == "initialize":
        from KGLMF.src.training.optimization import initialize
        print(initialize(args.config,args.manifest,args.output,args.seed))
    elif args.command == "feedback":
        from KGLMF.src.training.optimization import collect_feedback
        print(collect_feedback(args.config,args.manifest,args.checkpoint,args.output,args.seed,args.adapter))
    elif args.command == "score-feedback":
        from KGLMF.src.training.optimization import score_feedback
        print(score_feedback(args.config,args.manifest,args.feedback,args.output))
    elif args.command == "score-run":
        from KGLMF.src.training.optimization import score_predictions
        print(score_predictions(args.config,args.manifest,args.predictions,args.output))
    elif args.command == "export-memory":
        from KGLMF.src.training.optimization import export_memory
        print(export_memory(args.checkpoint,args.manifest,args.output))
    elif args.command == "finalize":
        from KGLMF.src.training.optimization import finalize
        print(finalize(args.config,args.manifest,args.checkpoint,args.feedback,args.output,args.seed))
    elif args.command == "run":
        from KGLMF.src.execution.runner import evaluate
        print(evaluate(args.config,args.manifest,args.checkpoint,args.suite,args.output,args.seed,args.conditions))
    else:
        from KGLMF.src.evaluation.reporting import aggregate, failure_sheet, export_harness, import_scores
        if args.command == "summarize":
            aggregate(args.runs,args.output,args.reference)
        elif args.command == "failures":
            failure_sheet(args.runs,args.output)
        elif args.command == "export-harness":
            export_harness(args.predictions,args.manifest,args.output)
        else:
            import_scores(args.predictions,args.scores,args.output)


if __name__ == "__main__":
    main()
