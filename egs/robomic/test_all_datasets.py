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
DISTURBANCE_NOISE_ARMSTRONG = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "noise" / "armstrong"
DISTURBANCE_NOISE_RED_HOT_CHILLIPEPPERS = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "noise" / "red_hot_chillipeppers"
DISTURBANCE_NOISE_SKRILLEX = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "noise" / "skrillex"
DISTURBANCE_NOISE_WHITE_NOISE = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "noise" / "white_noise"

DISTURBANCE_CUP_MIMICKING_M6x14 = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "robot_empty_cup-human_3_M6x14_cup" /"human_matches_robot_movements"
DISTURBANCE_CUP_SHAKING_M6x14 = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "robot_empty_cup-human_3_M6x14_cup" /"human_shakes_cup"

DISTURBANCE_CUP_MIMICKING_PLAYDOUGH = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "robot_empty_cup-human_playdough_cup" /"human_matches_robot_movements"
DISTURBANCE_CUP_SHAKING_PLAYDOUGH = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "robot_empty_cup-human_playdough_cup" /"human_shakes_cup"

DISTURBANCE_ROBOT_TAPPING = DATA_DIR / "icra2025-multiclass-v1-disturbances" / "tapping_robot"

# NOVEL_OBJECTS = DATA_DIR / "icra2025-new-objects"


LABELS_CSV_PATH = DATA_DIR / "robomic_categories.csv"

validation_sets = {
    "disturbance_noise_armstrong": DISTURBANCE_NOISE_ARMSTRONG,
    "disturbance_noise_red_hot_chillipeppers": DISTURBANCE_NOISE_RED_HOT_CHILLIPEPPERS,
    "disturbance_noise_skrillex": DISTURBANCE_NOISE_SKRILLEX,
    "disturbance_noise_white_noise": DISTURBANCE_NOISE_WHITE_NOISE,
    "disturbance_cup_mimicking_m6x14": DISTURBANCE_CUP_MIMICKING_M6x14,
    "disturbance_cup_shaking_m6x14": DISTURBANCE_CUP_SHAKING_M6x14,
    "disturbance_cup_mimicking_playdough": DISTURBANCE_CUP_MIMICKING_PLAYDOUGH,
    "disturbance_cup_shaking_playdough": DISTURBANCE_CUP_SHAKING_PLAYDOUGH,
    "disturbance_robot_tapping": DISTURBANCE_ROBOT_TAPPING,
    
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
    val_acc = stats[0]["acc"]
    confusion_matrix = stats[0]["confusion_matrix"]
    print(confusion_matrix)
    return val_acc, confusion_matrix

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



def main(mic_experiment_dir, laser_experiment_dir):

    checkpoint_name = "best_audio_model.pth"
    # checkpoint_name = "audio_model.11.pth"


    # find all fold folders, which start with "fold"
    mic_fold_model_dirs = [d for d in pathlib.Path(mic_experiment_dir).iterdir() if d.is_dir() and d.name.startswith("fold")]
    mic_models = [d / "models"/ checkpoint_name for d in mic_fold_model_dirs]

    laser_fold_model_dirs = [d for d in pathlib.Path(laser_experiment_dir).iterdir() if d.is_dir() and d.name.startswith("fold")]
    laser_models = [d / "models"/ checkpoint_name for d in laser_fold_model_dirs]

    models = {
        f"mic_fold_{i}": mic_models[i] for i in range(len(mic_models))
    }
    models.update({
        f"laser_fold_{i}": laser_models[i] for i in range(len(laser_models))
    })

    accuracies_dict = {}
    confusion_matrices_dict = {}

    for checkpoint in models:
        model = build_model(models[checkpoint])
        args = get_args_from_checkpoint(models[checkpoint])
        sensor_type = "mic" if "mic" in checkpoint else "laser"
        for validation_set_name in validation_sets:
            mic_dataset_json_path = get_validation_set_path(validation_set_name, sensor_type)
            val_acc, confusion_matrix = get_validation_accuracy(model, args, mic_dataset_json_path)

            if validation_set_name not in accuracies_dict:
                accuracies_dict[validation_set_name] = {}
            accuracies_dict[validation_set_name][checkpoint] = val_acc

            if validation_set_name not in confusion_matrices_dict:
                confusion_matrices_dict[validation_set_name] = {}
            confusion_matrices_dict[validation_set_name][checkpoint] = confusion_matrix


    
    # average the accuracies over the folds for each set and sensory type
    for validation_set_name in validation_sets:
        for sensor_type in ["mic", "laser"]:
            accuracies = [accuracies_dict[validation_set_name][f"{sensor_type}_fold_{i}"] for i in range(len(mic_models))]
            mean_accuracy = np.mean(accuracies)
            accuracies_dict[validation_set_name][sensor_type] = mean_accuracy
            lower_bound, upper_bound = wilson_score_interval(correct=np.sum(accuracies), n=len(accuracies))
            accuracies_dict[validation_set_name][f"{sensor_type}_lower_bound"] = lower_bound
            accuracies_dict[validation_set_name][f"{sensor_type}_upper_bound"] = upper_bound
            
    with open("validation_accuracies.json", "w") as f:
        json_dict = {"checkpoints": {"mic": mic_experiment_dir, "laser": laser_experiment_dir}, "accuracies": accuracies_dict}
        json.dump(json_dict, f, indent=4)


    # aggregate the confusion matrices by summing them
    for validation_set_name in validation_sets:
        for sensor_type in ["mic", "laser"]:
            confusion_matrices = [confusion_matrices_dict[validation_set_name][f"{sensor_type}_fold_{i}"] for i in range(len(mic_models))]
            confusion_matrix_sum = np.sum(confusion_matrices, axis=0)
            confusion_matrices_dict[validation_set_name][sensor_type] = confusion_matrix_sum


    # recursively crawl into the confusion matrix dict and convert all numpy arrays to lists
    def convert_numpy_to_list(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_numpy_to_list(value)
            elif isinstance(value, np.ndarray):
                d[key] = value.tolist()
    convert_numpy_to_list(confusion_matrices_dict)

    with open("validation_confusion_matrices.json", "w") as f:
        json_dict = {"checkpoints": {"mic": mic_experiment_dir, "laser": laser_experiment_dir}, "confusion_matrices": confusion_matrices_dict}
        json.dump(json_dict, f, indent=4)

if __name__ == "__main__":
    import argparse
    mic_experiment_dir =""
    laser_experiment_dir = ""
    
    # cli using argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mic", type=str, required=True, help="Path to the mic experiment")
    argparser.add_argument("--laser", type=str, required=True, help="Path to the laser experiment")
    args = argparser.parse_args()
    mic_experiment_dir = args.mic
    laser_experiment_dir = args.laser
    main(mic_experiment_dir, laser_experiment_dir)