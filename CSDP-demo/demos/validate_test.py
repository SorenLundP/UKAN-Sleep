#!/usr/bin/env python3
import sys
import os
import argparse
import importlib
import torch
import numpy as np # Keep numpy for the final summary table formatting if needed

# --- Path Setup ---
# Get the absolute path to the directory containing this script
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback for interactive sessions

# Go up two levels to get to the assumed LUMI_FILES directory
# !! Adjust if your directory structure is different !!
lumi_files_dir = os.path.dirname(os.path.dirname(script_dir))

# Add LUMI_FILES directory to the Python path for module imports
if lumi_files_dir not in sys.path:
    sys.path.insert(0, lumi_files_dir)

# Add the original project root for csdp_pipeline if it's not in LUMI_FILES
# Adjust this path if csdp_pipeline is located elsewhere relative to LUMI_FILES
ROOT_DIR = "/Users/sorenlund/Desktop/AU - Semester 6/Bachelor Projekt 2.0"
if ROOT_DIR not in sys.path:
    # Insert after LUMI_FILES to prioritize it if modules exist in both
    sys.path.insert(1, ROOT_DIR)

# Add the original USleep model directory (if different from LUMI_FILES)
# This assumes the 'usleep.py' from the first script is here:
usleep_orig_dir = os.path.join(
    ROOT_DIR, "ML architectures", "ml_architectures", "usleep"
)
if usleep_orig_dir not in sys.path:
    sys.path.insert(2, usleep_orig_dir)

importlib.invalidate_caches() # Good practice after path modification

# Print paths for debugging
print(f"--- Path Debugging ---")
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {script_dir}")
print(f"Assumed LUMI_FILES directory added to path: {lumi_files_dir}")
print(f"Original ROOT_DIR added to path: {ROOT_DIR}")
print(f"Original USleep dir added to path: {usleep_orig_dir}")
print(f"sys.path: {sys.path}")
print(f"--- End Path Debugging ---")


# --- Import necessary components AFTER path setup ---
try:
    # Import the KAN Gram model (used by most variants)
    from USleep_KAN_Gram.Usleep_KAN_Gram_BN import USleep_BottleneckGRAM

    # Import the original USleep model (used by one variant)
    # This assumes 'usleep.py' is findable via the paths set above
    from usleep import USleep

except ImportError as e:
    print(f"Error importing required model modules: {e}")
    print(
        "Ensure USleep_BottleneckGRAM and usleep are correctly installed/accessible."
    )
    print("Check the calculated paths and your project structure.")
    sys.exit(1)
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Could not find a specified module. Check:")
    print(
        "1. If 'USleep_KAN_Gram' directory exists within the LUMI_FILES directory."
    )
    print("2. If 'usleep.py' exists within the 'usleep_orig_dir'.")
    print("3. If the path calculations are correct for your structure.")
    sys.exit(1)


def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Count parameters for multiple USleep model variants."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=[
            "all",
            "original_usleep",
            "1%",
            "5%",
            "50%",
            "gram_sleep_train",
            "no_teach_gram_train",
            "normal_usleep_student_train",
            "train_script_UKAN",
        ],
        help="Model variant to count parameters for. Use 'all' for all variants.",
    )
    args = parser.parse_args()

    # --- Define Variant Configurations ---
    # Keys should match argparse choices. Weights info is ignored but kept for structure.
    variant_configs = {
        "original_usleep": {
            "model_class": USleep,
            "params": {
                "num_channels": 2,
                "initial_filters": 5,
                "complexity_factor": 0.5,
                "depth": 10,
                # "num_classes": 5, # Add if needed by USleep constructor
            },
            "weights": "Depth10_CF05.ckpt", # Ignored
            "weights_path_override": os.path.join( # Ignored
                usleep_orig_dir, "weights", "Depth10_CF05.ckpt"
            ),
        },
        "1%": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2,
                "initial_filters": 5,
                "complexity_factor": 0.16,
                "progression_factor": 1.4,
                "num_classes": 5,
                "depth": 10,
            },
            "weights": "final_UKAN_no_teacher_model_1%.pth", # Ignored
        },
        "5%": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2,
                "initial_filters": 5,
                "complexity_factor": 0.5,
                "progression_factor": 1.4,
                "num_classes": 5,
                "depth": 10,
            },
            "weights": "final_UKAN_no_teacher_model_5%.pth", # Ignored
        },
        "50%": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2,
                "initial_filters": 5,
                "complexity_factor": 3.9,
                "progression_factor": 1.4,
                "num_classes": 5,
                "depth": 11,
            },
            "weights": "final_UKAN_no_teacher_model_50%.pth", # Ignored
        },
        "gram_sleep_train": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2,
                "initial_filters": 5,
                "complexity_factor": 0.99,
                "progression_factor": 1.4,
                "num_classes": 5,
                "depth": 10,
            },
            "weights": "UKAN_no_teacher_model.pth", # Ignored
        },
        "no_teach_gram_train": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2,
                "initial_filters": 5,
                "complexity_factor": 0.99,
                "progression_factor": 1.4,
                "num_classes": 5,
                "depth": 10,
            },
            "weights": "final_student_model.pth", # Ignored
        },
        "normal_usleep_student_train": {
            "model_class": USleep,
            "params": {
                "num_channels": 2,
                "initial_filters": 3,
                "complexity_factor": 0.2,
                "progression_factor": 2,
                "depth": 10,
                # "num_classes": 5, # Add if needed by USleep constructor
            },
            "weights": "student_model_normal_usleep.pth", # Ignored
        },
        "train_script_UKAN": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2,
                "initial_filters": 5,
                "complexity_factor": 0.99,
                "progression_factor": 1.4,
                "num_classes": 5,
                "depth": 10,
            },
            "weights": "final_UKAN_no_teacher_model.pth", # Ignored
        },
    }

    # --- Determine Variants to Process ---
    if args.variant == "all":
        variants_to_process = list(variant_configs.keys())
    elif args.variant in variant_configs:
        variants_to_process = [args.variant]
    else:
        print(f"Error: Unknown variant '{args.variant}'.")
        print("Available choices:", list(variant_configs.keys()))
        sys.exit(1)

    # --- Device Setup (Optional for parameter counting, but harmless) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device (for potential future use): {device}")

    # --- Instantiate Each Selected Variant and Count Parameters ---
    results = {}
    for variant in variants_to_process:
        print("\n" + "=" * 80)
        print(f"Processing variant: {variant}")
        config = variant_configs[variant]

        # --- Instantiate Model ---
        print(f"Initializing model: {config['model_class'].__name__}")
        try:
            model_instance = config["model_class"](**config["params"])
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error initializing model for variant '{variant}': {e}")
            print(f"Parameters used: {config['params']}")
            results[variant] = {
                "params": None,
                "error": "Initialization failed",
            }
            continue # Skip to next variant

        # --- Count Parameters ---
        try:
            num_params = count_parameters(model_instance)
            results[variant] = {"params": num_params, "error": None}
            print(f"Trainable Parameters: {num_params:,}") # Format with commas
        except Exception as e:
            print(f"Error counting parameters for variant '{variant}': {e}")
            results[variant] = {"params": None, "error": "Param counting failed"}

    # --- Final Summary ---
    print("\n" + "=" * 80)
    print("Parameter Count Summary:")
    print("-" * 80)
    print(f"{'Variant':<30} | {'Trainable Parameters':<25} | {'Error'}")
    print("-" * 80)
    for variant, res in results.items():
        p_str = f"{res['params']:,}" if res["params"] is not None else "N/A"
        e_str = res["error"] if res["error"] else "None"
        print(f"{variant:<30} | {p_str:<25} | {e_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()
