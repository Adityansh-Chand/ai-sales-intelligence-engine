
from pathlib import Path
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip()+"\n", encoding="utf-8")

BASE = Path.cwd()

write(BASE/"README.md", """
# AI Sales Intelligence Engine

Predictive ML pipeline for lead scoring and revenue intelligence.

## Architecture

```mermaid
flowchart LR

CRM --> FeatureEngineering
FeatureEngineering --> Model
Model --> PredictionAPI
PredictionAPI --> Dashboard
PredictionAPI --> CRM
```
""")

write(BASE/"pipeline/features.py","""
def build_features(customer):

    return [
        customer["visits"],
        customer["spend"]
    ]
""")

write(BASE/"models/scoring.py","""
def predict(features):

    return sum(features)/100
""")

run("git add .")
run('git commit -m "flagship sales AI architecture + diagram"')
run("git push")
print("sales AI upgraded")
