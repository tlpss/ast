import pathlib
import torch 
import pandas as pd 
import numpy as np 
import json 
import sys 
import os
import pickle

path = pathlib.Path(__file__).parents[2] / "src"
# add to python path
sys.path.append(str(path))
from dataloader import AudiosetDataset
from models.ast_models import ASTModel
from traintest import validate

file_path = pathlib.Path(__file__)
DATA_DIR = file_path.parent / "data" 
DISTURBANCE_NOISE_ARMSTRONG = DATA_DIR / "icra2025-disturbances" / "noise" / "armstrong"
DISTURBANCE_NOISE_RED_HOT_CHILLIPEPPERS = DATA_DIR / "icra2025-disturbances" / "noise" / "red_hot_chillipeppers"
DISTURBANCE_NOISE_SKRILLEX = DATA_DIR / "icra2025-disturbances" / "noise" / "skrillex"
DISTURBANCE_NOISE_WHITE_NOISE = DATA_DIR / "icra2025-disturbances" / "noise" / "white_noise"

DISTURBANCE_CUP_MIMICKING = DATA_DIR / "icra2025-disturbances" / "robot_empty_cup-human_full_cup" /"human_matches_robot_movements"
DISTURBANCE_CUP_SHAKING = DATA_DIR / "icra2025-disturbances" / "robot_empty_cup-human_full_cup" /"human_shakes_cup"

NOVEL_OBJECTS = DATA_DIR / "icra2025-new-objects"

LABELS_CSV_PATH = DATA_DIR / "robomic_categories.csv"

validation_sets = {
    "disturbance_noise_armstrong": DISTURBANCE_NOISE_ARMSTRONG,
    "disturbance_noise_red_hot_chillipeppers": DISTURBANCE_NOISE_RED_HOT_CHILLIPEPPERS,
    "disturbance_noise_skrillex": DISTURBANCE_NOISE_SKRILLEX,
    "disturbance_noise_white_noise": DISTURBANCE_NOISE_WHITE_NOISE,
    "disturbance_cup_mimicking": DISTURBANCE_CUP_MIMICKING,
    "disturbance_cup_shaking": DISTURBANCE_CUP_SHAKING,
    "novel_objects": NOVEL_OBJECTS
}

def get_validation_set_path(validation_set_name, sensor_type):
    assert sensor_type in ["mic", "laser"], "sensor type must be either 'mic' or 'laser'"
    assert validation_set_name in validation_sets, f"unknown validation set {validation_set_name}"
    dataset_dir = validation_sets[validation_set_name]
    return dataset_dir / f"robomic_all_{sensor_type}.json"


def get_args_from_checkpoint(checkpoint: str) -> dict:
    pickled_args_file = pathlib.Path(checkpoint).parents[1] / "args.pkl"
    with open(pickled_args_file, "rb") as f:
        args = pickle.load(f)

        # legacy: for older checkpoints
        if not hasattr(args, "num_mel_bins"):
            args.num_mel_bins = 128
    
    return args


def build_model(checkpoint: str) -> ASTModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # these should match the model configuration during training! check  the train.sh file to ensure consistency

    # load args from checkpoint
    args = get_args_from_checkpoint(checkpoint)
    audio_model = ASTModel(label_dim=args.n_class, input_tdim=args.audio_length ,input_fdim=args.num_mel_bins)

    # mimick the train flow to load checkpoint.
    audio_model = torch.nn.DataParallel(audio_model)
    # load checkpoint
    checkpoint = torch.load(checkpoint, map_location=device)
    audio_model.load_state_dict(checkpoint)
    audio_model.to(device)
    audio_model.eval()
    return audio_model

def get_validation_accuracy(model, args, dataset_json_path) -> float:


    audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'mean': None, 'std': None, 'mode': 'validation', 'dataset': 'blabla'}

    audio_conf["mean"] = args.dataset_mean
    audio_conf["std"] = args.dataset_std
  
    print(f"using mean {audio_conf['mean']} and std {audio_conf['std']}, make sure these values match the training normalization")
    
    dataset = AudiosetDataset(dataset_json_path, label_csv=str(LABELS_CSV_PATH), audio_conf=audio_conf)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)

    # construct the args object (mimick the argparser in train.sh)
    args = {"loss_fn": torch.nn.CrossEntropyLoss(), "exp_dir": "/tmp/ast-robomic-eval"}
    # make keys accessible by dot notation
    args = type('obj', (object,), args)()
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    stats,_= validate(model, val_loader, args,"validation")
    val_acc = np.mean([stat["acc"] for stat in stats])
    return val_acc

def wilson_score_interval(correct, n, confidence=0.95):
    from scipy import stats as st
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



def main(mic_checkpoint, laser_checkpoint):
    mic_model = build_model(mic_checkpoint)
    laser_model = build_model(laser_checkpoint)

    mic_args = get_args_from_checkpoint(mic_checkpoint)
    laser_args = get_args_from_checkpoint(laser_checkpoint)

    accuracies_dict = {}
    for validation_set_name in validation_sets:
        mic_dataset_json_path = get_validation_set_path(validation_set_name, "mic")
        laser_dataset_json_path = get_validation_set_path(validation_set_name, "laser")
        mic_acc = get_validation_accuracy(mic_model, mic_args, mic_dataset_json_path)
        laser_acc = get_validation_accuracy(laser_model, laser_args, laser_dataset_json_path)
        print(f"Validation set: {validation_set_name}")
        print(f"Accuracy mic: {mic_acc}")
        print(f"Accuracy laser: {laser_acc}")
        print(f"---------------------------------")
        accuracies_dict[validation_set_name] = {"mic": mic_acc, "laser": laser_acc}


    
    with open("validation_accuracies.json", "w") as f:
        json_dict = {"checkpoints": {"mic": mic_checkpoint, "laser": laser_checkpoint}, "accuracies": accuracies_dict}
        json.dump(json_dict, f, indent=4)

if __name__ == "__main__":
    import argparse
    mic_checkpoint =""
    laser_checkpoint = ""
    
    # cli using argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mic", type=str, required=True, help="Path to the mic checkpoint")
    argparser.add_argument("--laser", type=str, required=True, help="Path to the laser checkpoint")
    args = argparser.parse_args()
    mic_checkpoint = args.mic
    laser_checkpoint = args.laser
    main(mic_checkpoint, laser_checkpoint)