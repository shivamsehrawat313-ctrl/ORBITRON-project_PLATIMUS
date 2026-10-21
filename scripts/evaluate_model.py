"""Compute binary metrics from two JSON arrays; requires real reference labels."""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.metrics import binary_metrics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("predictions",help="JSON file containing a binary prediction array")
parser.add_argument("references",help="JSON file containing the aligned binary reference array")
args=parser.parse_args()
pred=json.loads(Path(args.predictions).read_text()); ref=json.loads(Path(args.references).read_text())
print(json.dumps(binary_metrics(pred,ref),indent=2))
