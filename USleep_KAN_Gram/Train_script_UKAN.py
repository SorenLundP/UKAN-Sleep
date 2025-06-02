#!/usr/bin/env python3
import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import CohenKappa, F1Score

# Teacher model is imported as required:
from ml_architectures.ml_architectures.usleep.usleep import USleep

# Student network from your file.
from Usleep_KAN_Gram_BN import USleep_BottleneckGRAM

# Custom data loader and sampler utilities.
from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_pipeline.pipeline_elements.models import Split
from csdp_pipeline.pipeline_elements.samplers import (
    Random_Sampler,
    SamplerConfiguration,
)


# Global metric lists.
train_loss = []
train_acc = []
train_kappa = []
train_f1 = []
val_loss = []
val_acc = []
val_kappa = []
val_f1 = []


def collate_fn(batch):
    """
    Pads EEG and EOG signals to the maximum length in the batch.
    """
    eegs = [item["eeg"] for item in batch]
    eogs = [item["eog"] for item in batch]
    labels = [item["labels"] for item in batch]
    tags = [item["tag"] for item in batch]

    # Find the maximum length of EEG and EOG in the batch
    max_eeg_len = max([eeg.size(-1) for eeg in eegs])
    max_eog_len = max([eog.size(-1) for eog in eogs])

    # Pad EEG and EOG to the maximum length
    padded_eegs = [
        F.pad(eeg, (0, max_eeg_len - eeg.size(-1))) for eeg in eegs
    ]  # Pad at the end
    padded_eogs = [
        F.pad(eog, (0, max_eog_len - eog.size(-1))) for eog in eogs
    ]  # Pad at the end

    # Stack the padded tensors
    eegs = torch.stack(padded_eegs)
    eogs = torch.stack(padded_eogs)
    labels = torch.stack(labels)  # Assuming labels are already tensors

    print(f"EEG shape after padding: {eegs.shape}")
    print(f"EOG shape after padding: {eogs.shape}")

    return {"eeg": eegs, "eog": eogs, "labels": labels, "tag": tags}


def train_student_model(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    device,
    optimizer,
    scheduler,
    num_epochs=1000,
    student_weights_path="/users/srenlund/final_student_model.pth",
):
    """
    Trains the student model using knowledge distillation from the teacher.
    Training stops early if the validation accuracy does not improve for 150 epochs.
    Each time a new best validation accuracy is reached, the model is saved to
    student_weights_path.
    """
    best_val_acc = 0.0
    no_improvement_epochs = 0

    log_file_path = "training_log.txt"
    # Initialize the log file.
    with open(log_file_path, "w") as lf:
        lf.write("Training Log\n")

    for epoch in range(num_epochs):
        # ----------------- Training Phase -----------------
        student_model.train()
        epoch_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            # Assume each batch is a dictionary with keys: "eeg", "eog", and
            # "labels"
            eeg_signals = batch["eeg"].to(device)
            eog_signals = batch["eog"].to(device)
            labels = batch["labels"].to(device)

            # Concatenate EEG and EOG signals along the channel dimension.
            input_data = torch.cat([eeg_signals, eog_signals], dim=1)
            print(f"Input data shape before reshape: {input_data.shape}")

            # Reshape input_data to [batch_size, num_channels, sequence_length]
            input_data = input_data.reshape(
                input_data.shape[0], input_data.shape[1], -1
            )
            print(f"Input data shape after reshape: {input_data.shape}")

            optimizer.zero_grad()

            # Compute teacher predictions with no gradient.
            with torch.no_grad():
                teacher_logits = teacher_model(input_data.float())
                teacher_probs = F.softmax(teacher_logits, dim=1)

            # Compute student predictions.
            student_logits = student_model(input_data.float())
            student_probs = F.softmax(student_logits, dim=1)

            # Compute loss using KL divergence.
            student_log_probs = F.log_softmax(student_logits, dim=1)
            kl_loss = F.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean"
            )

            kl_loss.backward()
            optimizer.step()

            epoch_loss += kl_loss.item()

            # Collect predictions and targets.
            preds = student_probs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

            del eeg_signals, eog_signals, labels, input_data, teacher_logits, teacher_probs, student_logits, student_probs, student_log_probs, preds  # <-- ADD THIS

        # Average training loss over batches.
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_acc.append(acc)

        # Compute training Cohen Kappa and F1 score.
        unique_train = np.unique(train_targets)
        num_classes_train = len(unique_train)
        tr_kappa = CohenKappa(task="multiclass", num_classes=num_classes_train)(
            torch.tensor(train_preds), torch.tensor(train_targets)
        )
        train_kappa.append(tr_kappa.item())
        tr_f1 = F1Score(task="multiclass", num_classes=num_classes_train)(
            torch.tensor(train_preds), torch.tensor(train_targets)
        )
        train_f1.append(tr_f1.item())

        # ----------------- Validation Phase -----------------
        student_model.eval()
        val_epoch_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_eeg = val_batch["eeg"].to(device)
                val_eog = val_batch["eog"].to(device)
                val_labels = val_batch["labels"].to(device)

                val_input = torch.cat([val_eeg, val_eog], dim=1)

                # Reshape input_data to [batch_size, num_channels, sequence_length]
                val_input = val_input.reshape(
                    val_input.shape[0], val_input.shape[1], -1
                )

                val_logits = student_model(val_input.float())
                val_probs = F.softmax(val_logits, dim=1)

                # Use teacher predictions for the loss.
                teacher_logits = teacher_model(val_input.float())
                teacher_probs = F.softmax(teacher_logits, dim=1)
                batch_val_loss = F.kl_div(
                    F.log_softmax(val_logits, dim=1),
                    teacher_probs,
                    reduction="batchmean",
                )
                val_epoch_loss += batch_val_loss.item()

                preds = val_probs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(val_labels.cpu().numpy())

                del val_eeg, val_eog, val_labels, val_input, val_logits, val_probs, teacher_logits, teacher_probs, batch_val_loss, preds

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_loss.append(avg_val_loss)
        cur_val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_acc.append(cur_val_acc)
        unique_val = np.unique(val_targets)
        num_classes_val = len(unique_val)
        v_kappa = CohenKappa(task="multiclass", num_classes=num_classes_val)(
            torch.tensor(val_preds), torch.tensor(val_targets)
        )
        val_kappa.append(v_kappa.item())
        v_f1 = F1Score(task="multiclass", num_classes=num_classes_val)(
            torch.tensor(val_preds), torch.tensor(val_targets)
        )
        val_f1.append(v_f1.item())

        epoch_info = (
            f"Epoch {epoch+1:03d}: Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {acc:.4f}, Train Kappa: {tr_kappa.item():.4f}, "
            f"Train F1: {tr_f1.item():.4f} || Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {cur_val_acc:.4f}, Val Kappa: {v_kappa.item():.4f}, "
            f"Val F1: {v_f1.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        print(epoch_info)
        with open(log_file_path, "a") as lf:
            lf.write(epoch_info + "\n")

        # Update learning rate scheduler based on validation loss.
        scheduler.step(avg_val_loss)

        # Check for improvement in validation accuracy.
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            no_improvement_epochs = 0
            torch.save(student_model.state_dict(), student_weights_path)
            print("New best model saved!")
        else:
            no_improvement_epochs += 1

        # Early stopping after 150 epochs with no improvement.
        if no_improvement_epochs >= 150:
            print(
                "No improvement in validation accuracy for 150 consecutive epochs. Stopping training."
            )
            with open(log_file_path, "a") as lf:
                lf.write("Early stopping triggered.\n")
            break

    return student_model


def create_plots():
    plt.figure(figsize=(10, 6))

    # Plot losses.
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies.
    plt.subplot(2, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot Cohen Kappa.
    plt.subplot(2, 2, 3)
    plt.plot(train_kappa, label="Training Kappa")
    plt.plot(val_kappa, label="Validation Kappa")
    plt.title("Cohen Kappa")
    plt.xlabel("Epoch")
    plt.ylabel("Kappa")
    plt.legend()

    # Plot F1 score.
    plt.subplot(2, 2, 4)
    plt.plot(train_f1, label="Training F1")
    plt.plot(val_f1, label="Validation F1")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_plots():
    # Save the metrics and the generated figure.
    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_kappa": train_kappa,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_kappa": val_kappa,
        "val_f1": val_f1,
    }

    fig = plt.figure(figsize=(10, 6))
    create_plots()

    output_path = "/users/srenlund/training_metrics.pkl"
    # Save metrics and figure via pickle.
    with open(output_path, "wb") as f:
        pickle.dump(metrics, f)
        pickle.dump(fig, f)
    # Also save the figure as an image.
    fig.savefig("training_metrics_plot.png")
    print("Training metrics and plot saved to", output_path)
    print("Plot image saved as training_metrics_plot.png")


def main():
    # ------------------- Configuration -------------------
    print("####################################################################")
    print("Starting Training!!")
    batch_size = 16
    # Data and model weight paths.
    hdf5_file_path = "./data/shhs_spectro.hdf5"
    teacher_weights_path = (
        "./ml_architectures/ml_architectures/usleep/weights/Depth10_CF05.ckpt"
    )
    student_weights_path = "./final_student_model.pth"
    sleep_epochs_per_sample = 35

    num_train_batches = 3
    num_val_batches = 1

    # ------------------- Data Loading -------------------
    base_hdf5_path = os.path.dirname(hdf5_file_path)
    split = Split.random(
        base_hdf5_path=base_hdf5_path,
        split_name="train",
        split_percentages=(0.7, 0.2, 0.1),
    )
    train_sampler = Random_Sampler(
        split,
        num_epochs=sleep_epochs_per_sample,
        num_iterations=batch_size * num_train_batches,
    )
    val_sampler = Random_Sampler(
        split,
        num_epochs=sleep_epochs_per_sample,
        num_iterations=batch_size * num_val_batches,
    )
    samplers = SamplerConfiguration(
        train_sampler, val_sampler, None
    )  # Updated with validation sampler
    factory = Dataloader_Factory(training_batch_size=batch_size, samplers=samplers)
    train_loader = factory.training_loader(num_workers=1)
    val_loader = factory.validation_loader(num_workers=1)
    # Using the function collate_fn to pad eeg and eog to same size
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=collate_fn,
    )

    # ------------------- Model Setup -------------------
    # Initialize teacher model and load its weights.
    teacher_model = USleep(
        num_channels=2, initial_filters=5, complexity_factor=0.5, depth=10
    )
    teacher_model.load_state_dict(
        torch.load(teacher_weights_path, map_location="cpu")
    )
    teacher_model.eval()

    # Initialize student model.
    student_model = USleep_BottleneckGRAM(
        num_channels=2,
        initial_filters=5,
        complexity_factor=0.99,
        progression_factor=1.4,
        num_classes=5,
        depth=10,
    )

    # Xavier initialization for student model.
    def init_weights(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    student_model.apply(init_weights)

    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # ------------------- Optimizer & Scheduler -------------------
    optimizer = torch.optim.Adam(
        student_model.parameters(), lr=0.001, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    # ------------------- Training -------------------
    # Train using separate training and validation loaders.
    train_student_model(
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler,
        num_epochs=10000,
        student_weights_path=student_weights_path,
    )

    # Note: The best model has been saved during training.

    # ------------------- Plotting & Saving Metrics -------------------
    create_plots()
    save_plots()

    # Dump raw metrics lists so you can re-generate plots later.
    training_metrics_lists_path = "./training_metrics_lists.pkl"
    all_metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_kappa": train_kappa,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_kappa": val_kappa,
        "val_f1": val_f1,
    }
    with open(training_metrics_lists_path, "wb") as f:
        pickle.dump(all_metrics, f)
    print("Raw training metrics saved to", training_metrics_lists_path)


if __name__ == "__main__":
    main()

