#! /bin/bash
source venv/bin/activate
python Gan.py
cp ./captions_val2014_result_results.json ../Evaluation/coco-caption/results/
cd ../Evaluation/coco-caption/
source ./venv2/bin/activate
python2.7 main.py > metric_evaluation.txt