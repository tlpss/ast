import os
import sys 
import csv 
import json 
import pathlib

path = pathlib.Path(__file__).parents[2] / "src"
print(path)
# add to python path
sys.path.append(str(path))
import numpy as np 

from dataloader import AudiosetDataset
from models.ast_models import ASTModel
import torch
from traintest import validate
import argparse


# example command 
parser = argparse.ArgumentParser(description="Evaluate AST model on a validation set")
parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
parser.add_argument("--json", type=str, required=True, help="Path to the dataset JSON file")
parser.add_argument("--labels_csv", type=str, required=True, help="Path to the labels CSV file")

args = parser.parse_args()

model_checkpoint_path = args.ckpt
dataset_json_path = args.json
labels_csv = args.labels_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# these should match the model configuration during training! check  the train.sh file to ensure consistency
audio_model = ASTModel(label_dim=2,input_tdim=1100,input_fdim=128)

# mimick the train flow to load checkpoint.
audio_model = torch.nn.DataParallel(audio_model)
# load checkpoint
checkpoint = torch.load(model_checkpoint_path, map_location=device)
audio_model.load_state_dict(checkpoint)
audio_model.to(device)
audio_model.eval()

audio_conf = {'num_mel_bins': 128, 'target_length': 1100, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'mean': None, 'std': None, 'mode': 'train', 'dataset': 'blabla'}

if "mic" in dataset_json_path:
    audio_conf["mean"] = -5.44
    audio_conf["std"] = 4.28
elif "laser" in dataset_json_path:
    audio_conf["mean"] = -0.99
    audio_conf["std"] = 2.59
else:
    raise ValueError("Unknown normalization")
print(f"using mean {audio_conf['mean']} and std {audio_conf['std']}, make sure these values match the training normalization")

dataset = AudiosetDataset(dataset_json_path, label_csv=labels_csv, audio_conf=audio_conf)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)

# construct the args object (mimick the argparser in train.sh)
args = {"loss_fn": torch.nn.CrossEntropyLoss(), "exp_dir": "/tmp/ast-robomic-eval"}
if not os.path.exists(args["exp_dir"]):
    os.makedirs(args["exp_dir"])
# make keys accessible by dot notation
args = type('obj', (object,), args)()

stats, _ = validate(audio_model, val_loader, args, 'valid_set')

val_acc = stats[0]['acc']
val_mAUC = np.mean([stat['auc'] for stat in stats])
print('---------------evaluate on the validation set---------------')
print("Accuracy: {:.6f}".format(val_acc))
print("AUC: {:.6f}".format(val_mAUC))


# calculate a confidence interval using the Wilson score interval

import scipy.stats as st
def wilson_score_interval(correct, n, confidence=0.95):
    """
    Calculates the Wilson score confidence interval for a proportion.

    Args:
        correct (int): Number of correct observations.
        n (int): Total number of observations.
        confidence (float): Confidence level (e.g., 0.95 for 95% confidence).

    Returns:
        tuple: (lower_bound, upper_bound)
    """

    if n == 0:
        return (np.nan, np.nan)  # Handle the case where n is zero to prevent division by zero errors.

    p_hat = correct / n
    z = st.norm.ppf((1 + confidence) / 2)

    denominator = 1 + (z**2 / n)
    center_adjusted_probability = p_hat + (z**2 / (2 * n))
    adjusted_standard_deviation = z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))

    lower_bound = (center_adjusted_probability - adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + adjusted_standard_deviation) / denominator

    return (lower_bound, upper_bound)

n = len(dataset)
lower, upper = wilson_score_interval(int(val_acc * n), n)
print(f"95% confidence interval for accuracy: {lower:.6f} - {upper:.6f}")



predictions  = []
targets = []
for data in val_loader:
    inputs, target = data
    inputs = inputs.to(device)
    target = target.to(device)
    with torch.no_grad():
        output = audio_model(inputs)
    predictions.append(output)
    targets.append(target)
predictions = torch.cat(predictions).cpu().numpy()
targets = torch.cat(targets).cpu().numpy()

def calculate_using_bootstrapping():
    # calculate confidence interval using bootstrapping
    n_bootstraps = 10
    accuracies = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(targets), len(targets), replace=True)
        acc = np.mean(predictions[indices].argmax(axis=1) == targets[indices].argmax(axis=1))
        accuracies.append(acc)
    accuracies = np.array(accuracies)

    lower = np.percentile(accuracies, 2.5)
    upper = np.percentile(accuracies, 97.5)
    return lower, upper
lower, upper = calculate_using_bootstrapping()
print(f"95% confidence interval for accuracy using bootstrapping: {lower:.6f} - {upper:.6f}")