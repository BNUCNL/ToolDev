import os
import numpy as np
import pandas as pd
import json

# load config
f = open('config.json', 'r')
content = f.read()
config = json.loads(content)

# define params
out_path=config["out_path"]
support_path=config["support_path"]
dataset_path=config["dataset_path"]
model_path=config["model_path"]
clean_code=config["clean_code"]
normalize_metric=config["normalize_metric"]
