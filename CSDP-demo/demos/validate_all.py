#!/usr/bin/env python3
import sys
import os
import argparse
import importlib
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm
import h5py # Import h5py for direct reading

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
ROOT_DIR = "."
if ROOT_DIR not in sys.path:
    # Insert after LUMI_FILES to prioritize it if modules exist in both
    sys.path.insert(1, ROOT_DIR)

# Add the original USleep model directory (if different from LUMI_FILES)
# This assumes the 'usleep.py' from the first script is here:
usleep_orig_dir = os.path.join(ROOT_DIR, "ml_architectures", "usleep")
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
    from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
    # Import Split and Dataset_Split for manual creation
    from csdp_pipeline.pipeline_elements.models import Split, Dataset_Split #
    from csdp_pipeline.pipeline_elements.samplers import (
        Determ_sampler, # Use deterministic sampler for validation
        SamplerConfiguration,
        Random_Sampler # Needed for dummy sampler
    ) #
    # Import the KAN Gram model (used by most variants)
    from USleep_KAN_Gram.Usleep_KAN_Gram_BN import USleep_BottleneckGRAM #

    # Import the original USleep model (used by one variant)
    # This assumes 'usleep.py' is findable via the paths set above
    from usleep import USleep #

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure csdp_pipeline, USleep_BottleneckGRAM, and usleep are correctly installed/accessible.")
    print("Check the calculated paths and your project structure.")
    sys.exit(1)
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Could not find a specified module. Check:")
    print("1. If 'USleep_KAN_Gram' directory exists within the LUMI_FILES directory.")
    print("2. If 'usleep.py' exists within the 'usleep_orig_dir'.")
    print("3. If the path calculations are correct for your structure.")
    sys.exit(1)


def evaluate_model(model, val_loader, device):
    """
    Run the evaluation loop using class index labels and return metrics.
    """
    model.eval()
    all_predictions = []
    all_true_labels = []

    print("Starting validation for current variant...")
    with torch.no_grad():
        # Use tqdm for progress bar
        for batch in tqdm(val_loader, desc="Validation Batches", leave=False): # Added leave=False
            try:
                eeg_signals = batch["eeg"].to(device)
                eog_signals = batch["eog"].to(device)
                # Assume labels are class indices
                true_labels = batch["labels"].to(device).long() # Ensure Long type

            except KeyError as e:
                print(f"Error: Missing key in batch data: {e}. Check DataLoader.")
                continue
            except Exception as e:
                print(f"Error processing batch data: {e}")
                continue

            # Concatenate inputs
            input_data = torch.cat([eeg_signals, eog_signals], dim=1).float()

            # Get model output (logits)
            try:
                output_logits = model(input_data)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")
                continue # Skip batch if forward pass fails

            # Get predictions
            batch_predictions = torch.argmax(output_logits, dim=1)

            # Sanity Checks
            if true_labels.numel() == 0 or batch_predictions.numel() == 0:
                print("Warning: Skipping batch with empty labels or predictions.")
                continue
            if batch_predictions.shape != true_labels.shape:
                print(f"Warning: Shape mismatch! Preds: {batch_predictions.shape}, Labels: {true_labels.shape}. Skipping batch.")
                print(f"Input shape to model: {input_data.shape}")
                print(f"Output logits shape: {output_logits.shape}")
                continue

            # Store Results (Flattened)
            all_predictions.extend(batch_predictions.view(-1).cpu().numpy())
            all_true_labels.extend(true_labels.view(-1).cpu().numpy())

    # --- Calculate Metrics ---
    if not all_true_labels or not all_predictions:
        print("No valid data processed. Cannot calculate metrics.")
        return None, None # Return None if no data

    final_predictions = np.array(all_predictions)
    final_true_labels = np.array(all_true_labels)

    if len(final_predictions) != len(final_true_labels):
         print(f"Error: Mismatch in total number of predictions ({len(final_predictions)}) and labels ({len(final_true_labels)}).")
         return None, None # Return None on mismatch

    print(f"Total Timesteps Evaluated: {len(final_true_labels)}")

    # Calculate Accuracy
    try:
        accuracy = accuracy_score(final_true_labels, final_predictions)
    except Exception as e:
        print(f"Error calculating Accuracy: {e}")
        accuracy = None

    # Calculate Cohen's Kappa
    kappa = None
    unique_labels = np.unique(final_true_labels)
    unique_preds = np.unique(final_predictions)
    print(f"Unique True Labels: {unique_labels}")
    print(f"Unique Predictions: {unique_preds}")

    if len(unique_labels) <= 1:
        print("Warning: Only one class present in true labels. Kappa is undefined.")
        kappa = float('nan')
    elif len(unique_preds) <= 1 and len(unique_labels) > 1:
         print("Warning: Only one class predicted, but multiple true classes exist. Kappa might be misleading or 0.")
         try:
             kappa = cohen_kappa_score(final_true_labels, final_predictions)
         except Exception as e:
             print(f"Error calculating Cohen's Kappa: {e}")
             kappa = float('nan')
    else:
        try:
            kappa = cohen_kappa_score(final_true_labels, final_predictions)
        except Exception as e:
            print(f"Error calculating Cohen's Kappa: {e}")
            kappa = float('nan')

    return kappa, accuracy


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Evaluate multiple USleep model variants."
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
        help="Model variant to evaluate. Use 'all' to evaluate all variants.",
    )
    args = parser.parse_args()

    # --- Define Variant Configurations ---
    variant_configs = {
         "Original USleep": {
            "model_class": USleep,
            "params": {
                "num_channels": 2, "initial_filters": 5, "complexity_factor": 0.5, "depth": 10,
            },
            "weights": "Depth10_CF05.ckpt",
            "weights_path_override": os.path.join(
                 usleep_orig_dir, "weights", "Depth10_CF05.ckpt"
            )
        },
        "UKAN-Sleep 1% (No Teacher)": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2, "initial_filters": 5, "complexity_factor": 0.16,
                "progression_factor": 1.4, "num_classes": 5, "depth": 10,
            },
            "weights": "final_UKAN_no_teacher_model_1%.pth",
        },
        "UKAN-Sleep 5% (No Teacher)": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2, "initial_filters": 5, "complexity_factor": 0.5,
                "progression_factor": 1.4, "num_classes": 5, "depth": 10,
            },
            "weights": "final_UKAN_no_teacher_model_5%.pth",
        },
        "UKAN-Sleep 50% (No Teacher)": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2, "initial_filters": 5, "complexity_factor": 3.9,
                "progression_factor": 1.4, "num_classes": 5, "depth": 11,
            },
            "weights": "final_UKAN_no_teacher_model_50%.pth",
        },
        "UKAN-Sleep 10% (Student)": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2, "initial_filters": 5, "complexity_factor": 0.99,
                "progression_factor": 1.4, "num_classes": 5, "depth": 10,
            },
            "weights": "UKAN_no_teacher_model.pth",
        },
        "UKAN-Sleep 10% (No Teacher)": {
            "model_class": USleep_BottleneckGRAM,
            "params": {
                "num_channels": 2, "initial_filters": 5, "complexity_factor": 0.99,
                "progression_factor": 1.4, "num_classes": 5, "depth": 10,
            },
            "weights": "final_student_model.pth",
        },
        "USleep 10% (Student)": {
            "model_class": USleep,
            "params": {
                "num_channels": 2, "initial_filters": 3, "complexity_factor": 0.2,
                "progression_factor": 2, "depth": 10,
            },
            "weights": "student_model_normal_usleep.pth",
        }
    }

    # --- Determine Variants to Test ---
    if args.variant == "all":
        variants_to_test = list(variant_configs.keys())
    elif args.variant in variant_configs:
        variants_to_test = [args.variant]
    else:
        print(f"Error: Unknown variant '{args.variant}'.")
        print("Available choices:", list(variant_configs.keys()))
        sys.exit(1)

    # --- Basic configuration ---
    batch_size = 1
    # !! POINT THIS TO YOUR HDF5 FILE !!
    #hdf5_file_path = "/Users/sorenlund/Downloads/dod-o.hdf5"
    hdf5_file_path = "/Users/sorenlund/Downloads/svuh.hdf5"
    if not os.path.exists(hdf5_file_path):
        print(f"ERROR: HDF5 file not found at: {hdf5_file_path}")
        print("Please update the hdf5_file_path variable in the script.")
        sys.exit(1)

    base_hdf5_path = os.path.dirname(hdf5_file_path)
    dataloader_workers = 8

    # --- Manually Create the Split Object ---
    print("Manually creating Split object...")
    split_id = "manual_eval_split"
    subject_keys = []
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # --- Modified Logic: Read keys from the 'data' group --- #
            if 'data' in f:
                data_group = f['data']
                subject_keys = list(data_group.keys())
            else:
                # Fallback or error if 'data' group is missing
                print(f"ERROR: '/data' group not found in HDF5 file: {hdf5_file_path}")
                print("Cannot determine subject keys. Please check HDF5 structure.")
                sys.exit(1) # Exit if the expected structure isn't found

        if not subject_keys:
             raise ValueError("No subject keys found within the '/data' group in the HDF5 file.")

        print(f"Found subjects under '/data': {subject_keys}") # Modified print statement

        # Create a Dataset_Split where all subjects are in the 'test' list
        dataset_split_info = Dataset_Split(
            dataset_filepath=hdf5_file_path,
            train=[],
            val=[],
            test=subject_keys # Assign all subjects to the test set
        )

        # Create the main Split object
        split = Split(
            id=split_id,
            dataset_splits=[dataset_split_info],
            base_data_path=base_hdf5_path
        )
        print("Manual Split object created successfully.")

    except Exception as e:
        print(f"Error reading HDF5 file or creating manual split: {e}")
        print(f"Please check the HDF5 file path and its internal structure: {hdf5_file_path}")
        sys.exit(1)
    # --- End Manual Split Creation ---


    # --- Define Samplers and Dataloaders (using the manual split) ---
    print("Setting up samplers and dataloader using manual split...")
    try:
        # Use the 'test' split for the deterministic sampler
        test_sampler = Determ_sampler(split, split_type="test") # Use test split
        # Dummy sampler for train (not used but required by SamplerConfiguration)
        dummy_train_sampler = Random_Sampler(split, num_epochs=1, num_iterations=1)
        # Dummy sampler for val (not used but required by SamplerConfiguration)
        dummy_val_sampler = Determ_sampler(split, split_type="val") # Val split is empty

        samplers = SamplerConfiguration(dummy_train_sampler, dummy_val_sampler, test_sampler) # Put test_sampler in correct slot
        factory = Dataloader_Factory(batch_size, samplers)
        # Use the 'testing_loader' which corresponds to the 'test' sampler
        eval_loader = factory.testing_loader(num_workers=dataloader_workers)
        print("Samplers and Dataloader setup complete.")
    except Exception as e:
        print(f"Error setting up samplers/dataloader: {e}")
        sys.exit(1)

    if eval_loader is None:
         print("Error: Evaluation loader could not be created.")
         sys.exit(1)
    if len(eval_loader) == 0:
         print("Warning: Evaluation data loader is empty. Check HDF5 file and subject keys.")
         # Exiting because evaluation is not possible
         sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Evaluate Each Selected Variant ---
    results = {}
    for variant in variants_to_test:
        print("\n" + "=" * 80)
        print(f"Evaluating variant: {variant}")
        config = variant_configs[variant]

        # --- Instantiate Model ---
        print(f"Initializing model: {config['model_class'].__name__}")
        try:
            model_instance = config["model_class"](**config["params"])
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error initializing model for variant '{variant}': {e}")
            print(f"Parameters used: {config['params']}")
            results[variant] = {"kappa": None, "accuracy": None, "error": "Initialization failed"}
            continue

        # --- Load Weights ---
        weight_filename = config["weights"]
        if "weights_path_override" in config and config["weights_path_override"]:
             model_path = config["weights_path_override"]
             print(f"Using overridden weight path: {model_path}")
        else:
             model_path = os.path.join(lumi_files_dir, weight_filename)
             print(f"Looking for weights at: {model_path}")


        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                if all(key.startswith('model.') for key in state_dict.keys()):
                    state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}
                elif any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
                model_instance.load_state_dict(state_dict, strict=False)
                print(f"Model weights loaded successfully from {model_path}.")
            except Exception as e:
                print(f"Error loading weights for variant '{variant}': {e}")
                results[variant] = {"kappa": None, "accuracy": None, "error": f"Weight loading failed: {e}"}
                continue
        else:
            print(f"Error: Model weights file not found: {model_path}")
            results[variant] = {"kappa": None, "accuracy": None, "error": "Weight file not found"}
            continue

        model_instance.to(device)

        # --- Run Evaluation ---
        kappa, accuracy = evaluate_model(model_instance, eval_loader, device)

        # --- Store and Print Results ---
        results[variant] = {"kappa": kappa, "accuracy": accuracy, "error": None}
        if kappa is not None and accuracy is not None:
            print(f"\n--- Results for Variant '{variant}' ---")
            print(f"  Cohen's Kappa: {kappa:.4f}")
            print(f"  Accuracy:      {accuracy:.4f}")
            print("-" * 40)
        else:
            print(f"\n--- Evaluation failed or produced no results for Variant '{variant}' ---")

    # --- Final Summary ---
    print("\n" + "=" * 80)
    print("Evaluation Complete. Summary:")
    print("-" * 80)
    print(f"{'Variant':<30} | {'Kappa':<15} | {'Accuracy':<15} | {'Error'}")
    print("-" * 80)
    for variant, res in results.items():
        k_str = f"{res['kappa']:.4f}" if res['kappa'] is not None and not np.isnan(res['kappa']) else "N/A"
        a_str = f"{res['accuracy']:.4f}" if res['accuracy'] is not None else "N/A"
        e_str = res['error'] if res['error'] else "None"
        print(f"{variant:<30} | {k_str:<15} | {a_str:<15} | {e_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()