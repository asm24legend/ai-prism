"""Generate the hyperparameter table from the config that was actually used (never hand-edit)."""
import sys
import yaml
from src.config import DEFAULTS

cfg = {**DEFAULTS, **(yaml.safe_load(open(sys.argv[1])) or {})}
print(f"| setting | value |\n|---|---|")
for k, v in cfg.items():
    print(f"| {k} | {v} |")
