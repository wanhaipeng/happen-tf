#!/bin/bash
python3 deep_sort_app.py \
  --sequence_dir=./MOT16/test/MOT16-14 \
  --detection_file=./resources/detections/MOT16_POI_test/MOT16-14.npy \
  --min_confidence=0.3 \
  --nn_budget=100 \
  --display=True
