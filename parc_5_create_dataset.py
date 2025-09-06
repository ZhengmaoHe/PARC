from pathlib import Path
import yaml
from PARC.util.create_dataset import create_dataset_yaml
"""
Takes folders of motion data and creates a dataset .yaml file with proporitional sampling weights.
Motion classes are created based on the first layer of nested folders.
By default, all motion classes will have the same weighting: 1.0.
Optionally also computes processed terrain data.
"""

with open("PARC/create_dataset_config.yaml", "r") as f:
    config = yaml.safe_load(f)

