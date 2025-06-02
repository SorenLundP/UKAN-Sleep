import sys
import os
import importlib
import torch
import numpy as np
# Use sklearn metrics as in the original script
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F # Keep for potential debugging/softmax inspection

# --- Path Setup (Relative to script location) ---
# Get the absolute path to the directory containing this script
# Use a placeholder if __file__ is not defined (e.g., interactive session)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback to current working directory
# Go up two levels to get to the assumed LUMI_FILES directory
# !! Adjust this if your directory structure is different !!
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

# Add the USleep model directory (original one, might not be needed now)
usleep_dir = os.path.join(ROOT_DIR, "ML architectures", "ml_architectures", "usleep")
if usleep_dir not in sys.path:
    sys.path.insert(2, usleep_dir)

importlib.invalidate_caches() # Good practice after path modification

# Print paths for debugging
print(f"--- Path Debugging ---")
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {script_dir}")
print(f"Assumed LUMI_FILES directory added to path: {lumi_files_dir}")
print(f"Original ROOT_DIR added to path: {ROOT_DIR}")
print(f"sys.path: {sys.path}")
print(f"--- End Path Debugging ---")


# --- Import necessary components AFTER path setup ---
try:
    from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
    from csdp_pipeline.pipeline_elements.models import Split
    from csdp_pipeline.pipeline_elements.samplers import (
        Determ_sampler, # Use deterministic sampler for validation
        SamplerConfiguration,
        Random_Sampler # Needed for dummy sampler in original setup
    )
    # Import the new model
    from USleep_KAN_Gram.Usleep_KAN_Gram_BN import USleep_BottleneckGRAM
    # from usleep import USleep # Keep commented out or remove if not needed
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure csdp_pipeline and USleep_BottleneckGRAM are correctly installed/accessible.")
    print("Check the calculated paths and your project structure.")
    sys.exit(1)
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Could not find the specified module. Check:")
    print("1. If 'USleep_KAN_Gram' directory exists within the LUMI_FILES directory.")
    print("2. If 'Usleep_KAN_Gram_BN.py' exists within 'USleep_KAN_Gram'.")
    print("3. If the LUMI_FILES path calculation is correct for your structure.")
    sys.exit(1)


def validate_model(
    model_path: str,
    # Removed arguments that are now defined inside, matching original script
) -> None:
    # --- Configuration (matching original script, adjust if needed) ---
    batch_size = 1 # Consider increasing for faster validation if memory allows
    # !! Ensure this HDF5 path is correct relative to where you run !!
    # It might be better to define it relative to LUMI_FILES or ROOT_DIR
    hdf5_file_path = os.path.join(ROOT_DIR, "data", "abc.hdf5") # Using original ROOT_DIR
    base_hdf5_path = os.path.dirname(hdf5_file_path)
    # Use current working directory for split file, as in original script
    split_file_path = os.getcwd()
    dataloader_workers = 8
    split_json_path = os.path.join(split_file_path, "demo.json") # Saving split in CWD

    print(f"Looking for split file at: {split_json_path}")
    print(f"Using base HDF5 path: {base_hdf5_path}")

    # --- Use or generate the split file (original logic) ---
    if not os.path.exists(split_json_path):
        print(f"Split file not found. Generating new split 'demo.json' in {split_file_path}...")
        try:
            # Ensure the target directory exists if split_file_path is nested
            os.makedirs(os.path.dirname(split_json_path), exist_ok=True)
            split = Split.random(
                base_hdf5_path=base_hdf5_path,
                split_name="demo",
                # Using small validation split for demo purposes
                split_percentages=(0.01, 0.01, 1-0.02), # Train 1%, Val 1%, Test 98%
            )
            split.dump_file(path=split_file_path) # Save to cwd
            print("Split file generated.")
        except Exception as e:
            print(f"Error generating split file: {e}")
            return
    else:
        print("Using existing split file.")
        try:
            split = Split.file(split_json_path)
        except Exception as e:
            print(f"Error loading split file {split_json_path}: {e}")
            return

    # --- Setup Samplers and DataLoader (original logic for validation) ---
    print("Setting up sampler for 'val' split...")
    try:
        # Use deterministic sampler for validation consistency
        val_sampler = Determ_sampler(split, split_type="val")

        # Create dummy train sampler as required by original SamplerConfiguration setup
        dummy_train_sampler = Random_Sampler(split, num_epochs=1, num_iterations=1)

        samplers = SamplerConfiguration(dummy_train_sampler, val_sampler, None)
        factory = Dataloader_Factory(batch_size, samplers)
        val_loader = factory.validation_loader(num_workers=dataloader_workers)
    except TypeError as e:
        print(f"TypeError during sampler/dataloader setup: {e}")
        print("Check the arguments passed to the Sampler constructors.")
        return
    except Exception as e:
        print(f"Error setting up samplers/dataloader: {e}")
        return

    if val_loader is None:
         print("Error: Validation loader could not be created.")
         return
    if len(val_loader) == 0:
         print("Warning: Validation data loader is empty. Check split percentages and data.")
         return # Exit if no validation data

    # --- Setup Model and Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize the NEW Model ---
    print("Initializing USleep_BottleneckGRAM model...")
    try:
        model = USleep_BottleneckGRAM(
            num_channels=2,       # From training script
            initial_filters=5,    # From training script
            complexity_factor=0.16, # From training script
            progression_factor=1.4, # From training script
            num_classes=5,        # From training script (Ensure this matches labels)
            depth=10              # From training script
        )
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing USleep_BottleneckGRAM model: {e}")
        return

    # --- Load Model Weights ---
    print(f"Attempting to load model weights from: {model_path}")
    if os.path.exists(model_path):
        try:
            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=device)

            # --- Handle potential keys mismatch (e.g., 'model.' prefix) ---
            # Check if keys start with 'model.' (common in some saving methods)
            if all(key.startswith('model.') for key in state_dict.keys()):
                print("Detected 'model.' prefix in state_dict keys. Removing prefix.")
                state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}
            elif any(key.startswith('module.') for key in state_dict.keys()):
                print("Detected 'module.' prefix (likely from DataParallel). Removing prefix.")
                state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}


            # Load the adjusted state dictionary
            model.load_state_dict(state_dict)
            print(f"Model weights loaded successfully from {model_path}.")
        except FileNotFoundError:
            # This check is technically redundant due to os.path.exists, but good practice
            print(f"Error: Model file not found at {model_path}")
            return
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Potential issues:")
            print("1. Model architecture in script doesn't match saved weights.")
            print("2. The .pth file might be corrupted or saved incorrectly.")
            print("3. Keys mismatch (e.g., missing 'model.' prefix handling).")
            # You might want to print keys for debugging:
            # print("Model keys:", model.state_dict().keys())
            # print("Loaded keys:", state_dict.keys())
            return
        except Exception as e:
            print(f"An unexpected error occurred loading the model: {e}")
            return
    else:
        print(f"Error: Model weights file not found at the specified path: {model_path}")
        return

    model.to(device)
    model.eval() # Set to evaluation mode

    # --- Validation Loop (Corrected Label Handling) ---
    all_predictions = []
    all_true_labels = []

    print("Starting validation on 'val' split...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation Batches"):
            try:
                eeg_signals = batch["eeg"].to(device)
                eog_signals = batch["eog"].to(device)
                true_labels = batch["labels"].to(device).long() # Ensure Long type

            except KeyError as e:
                print(f"Error: Missing key in batch data: {e}. Check DataLoader.")
                continue
            except Exception as e:
                print(f"Error processing batch data: {e}")
                continue

            # Concatenate inputs
            # Ensure float type, model might expect float32
            input_data = torch.cat([eeg_signals, eog_signals], dim=1).float()

            # Get model output (logits)
            # Shape: [batch_size, num_classes, seq_len] (expected)
            try:
                output_logits = model(input_data)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")
                continue # Skip batch if forward pass fails

            # Get predictions by taking argmax over class dimension
            # Shape: [batch_size, seq_len] (expected)
            batch_predictions = torch.argmax(output_logits, dim=1)

            # --- Sanity Checks ---
            if true_labels.numel() == 0 or batch_predictions.numel() == 0:
                print("Warning: Skipping batch with empty labels or predictions.")
                continue
            if batch_predictions.shape != true_labels.shape:
                print(f"Warning: Shape mismatch! Preds: {batch_predictions.shape}, Labels: {true_labels.shape}. Skipping batch.")
                # Debug shapes if mismatch occurs
                print(f"Input shape to model: {input_data.shape}")
                print(f"Output logits shape: {output_logits.shape}")
                continue

            # --- Store Results (Flattened) ---
            all_predictions.extend(batch_predictions.view(-1).cpu().numpy())
            all_true_labels.extend(true_labels.view(-1).cpu().numpy())

    # --- Calculate Metrics ---
    if not all_true_labels or not all_predictions:
        print("No valid data processed. Cannot calculate metrics.")
        return

    final_predictions = np.array(all_predictions)
    final_true_labels = np.array(all_true_labels)

    if len(final_predictions) != len(final_true_labels):
         print(f"Error: Mismatch in total number of predictions ({len(final_predictions)}) and labels ({len(final_true_labels)}).")
         return

    print("\n--- Validation Results ---")
    print(f"Model: USleep_BottleneckGRAM")
    print(f"Weights: {os.path.basename(model_path)}")
    print(f"Data Split Evaluated: 'val'")
    print(f"Total Timesteps Evaluated: {len(final_true_labels)}")

    # Calculate Accuracy
    try:
        accuracy = accuracy_score(final_true_labels, final_predictions)
        print(f"Overall Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error calculating Accuracy: {e}")

    # Calculate Cohen's Kappa
    # Ensure there are multiple classes for Kappa calculation
    unique_labels = np.unique(final_true_labels)
    unique_preds = np.unique(final_predictions)
    print(f"Unique True Labels: {unique_labels}")
    print(f"Unique Predictions: {unique_preds}")

    if len(unique_labels) <= 1:
        print("Warning: Only one class present in true labels. Kappa is undefined.")
        kappa = float('nan') # Assign NaN if Kappa is undefined
    elif len(unique_preds) <= 1 and len(unique_labels) > 1:
         print("Warning: Only one class predicted, but multiple true classes exist. Kappa might be misleading or 0.")
         try:
             kappa = cohen_kappa_score(final_true_labels, final_predictions)
             print(f"Cohen's Kappa Score: {kappa:.4f}")
         except Exception as e:
             print(f"Error calculating Cohen's Kappa: {e}")
             kappa = float('nan')
    else:
        try:
            kappa = cohen_kappa_score(final_true_labels, final_predictions)
            print(f"Cohen's Kappa Score: {kappa:.4f}")
        except Exception as e:
            print(f"Error calculating Cohen's Kappa: {e}")
            kappa = float('nan') # Assign NaN on error

    print("--------------------------")


def main() -> None:
    # --- Configuration ---
    # Define the relative path to the model weights file
    model_weights_filename = "final_UKAN_no_teacher_model_1%.pth"
    # Construct the full path using the calculated LUMI_FILES directory
    # !! Ensure 'lumi_files_dir' is calculated correctly above !!
    model_path = os.path.join(lumi_files_dir, model_weights_filename)

    print(f"Running validation for model: {model_path}")

    # --- Run Validation ---
    validate_model(model_path=model_path)


if __name__ == "__main__":
    main()
