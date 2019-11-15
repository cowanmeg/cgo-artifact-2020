#!/bin/bash

# Generate graphs from data

python3 scripts/graph_speedup_over_fp.py
python3 scripts/graph_speedup_over_pytorch.py

python3 scripts/graph_accuracy.py
python3 scripts/graph_end2end.py

python3 scripts/heatmap_data.py
