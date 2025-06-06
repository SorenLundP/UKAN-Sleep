#!/usr/bin/env python3
import os
import sys
import pickle
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Teacher model is imported as required:
from ml_architectures.ml_architectures.usleep.usleep import USleep

# Custom data loader and sampler utilities.
from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_pipeline.pipeline_elements.samplers import Random_Sampler, SamplerConfiguration
from csdp_pipeline.pipeline_elements.models import Split

# Student network from your file.
from Usleep_KAN_Gram_BN import USleep_BottleneckGRAM

# Additional helpers.
from torchmetrics import CohenKappa, F1Score
from torch.utils.data import DataLoader

# Initialize metrics
train_loss = [] 
train_acc = []
train_kappa = []
train_f1 = []
val_loss = []
val_acc = []
val_kappa = []
val_f1 = []

def compute_metrics(preds, targets, num_classes):
    preds = preds.argmax(dim=1)
    targets = targets.argmax(dim=1)
    
    accuracy = torch.sum(preds == targets) / len(targets)
    kappa = CohenKappa(task="multiclass", num_classes=num_classes)(preds, targets)
    f1 = F1Score(num_classes=num_classes)(preds, targets)
    
    return accuracy.item(), kappa.item(), f1.item()

def train_student_model(teacher_model, student_model, data_loader, device, optimizer, scheduler, num_epochs=30000000):
    global train_loss, train_acc, train_kappa, train_f1, val_loss, val_acc, val_kappa, val_f1
    
    best_val_acc = 0.0
    no_improvement_epochs = 0
    log_file = open('training_log_normal_usleep.txt', 'w')
    log_file.write('timestamp,epoch,train_loss,train_acc,train_kappa,train_f1,val_loss,val_acc,val_kappa,val_f1,lr\n')
    log_file.close()
    
    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in data_loader:
            eeg_signals = batch["eeg"].to(device)
            eog_signals = batch["eog"].to(device)
            labels = batch["labels"].to(device)
            
            input_data = torch.cat([eeg_signals, eog_signals], dim=1)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher_model(input_data.float())
                teacher_probs = F.softmax(teacher_logits, dim=1)
            
            student_logits = student_model(input_data.float())
            student_probs = F.softmax(student_logits, dim=1)
            
            student_log_probs = F.log_softmax(student_logits, dim=1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            
            kl_loss.backward()
            optimizer.step()
            
            epoch_loss += kl_loss.item()
            
            # Collect predictions and targets for metrics
            preds = student_probs.argmax(dim=1)
            targets = labels
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate average loss
        avg_loss = epoch_loss / len(data_loader)
        train_loss.append(avg_loss)
        
        # Calculate train metrics
        acc = np.mean(np.array(all_preds) == np.array(all_targets))
        train_acc.append(acc)
        
        # Calculate Kappa
        unique_targets = np.unique(all_targets)
        num_classes = len(unique_targets)
        kappa = CohenKappa(task="multiclass", num_classes=num_classes)(torch.tensor(all_preds), torch.tensor(all_targets))
        train_kappa.append(kappa.item())
        
        # Calculate F1 score
        f1 = F1Score(task="multiclass", num_classes=num_classes)(torch.tensor(all_preds), torch.tensor(all_targets))
        train_f1.append(f1.item())
        
        # Validation
        if epoch % 2 == 0:  # Validate every 2 epochs
            student_model.eval()
            val_epoch_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for val_batch in data_loader:
                    val_eeg = val_batch["eeg"].to(device)
                    val_eog = val_batch["eog"].to(device)
                    val_labels = val_batch["labels"].to(device)

                    val_input = torch.cat([val_eeg, val_eog], dim=1)
                    val_logits = student_model(val_input.float())
                    val_probs = F.softmax(val_logits, dim=1)

                    # Get teacher predictions
                    teacher_logits = teacher_model(val_input.float())
                    teacher_probs = F.softmax(teacher_logits, dim=1)

                    # Calculate KL divergence loss like in training
                    batch_val_loss = F.kl_div(F.log_softmax(val_logits, dim=1),
                                              teacher_probs,
                                              reduction='batchmean')
                    val_epoch_loss += batch_val_loss.item()

                    preds = val_probs.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(val_labels.cpu().numpy())

                avg_val_loss = val_epoch_loss / len(data_loader)
                # Append average loss to the global val_loss list
                val_loss.append(avg_val_loss)

                acc = np.mean(np.array(val_preds) == np.array(val_targets))
                val_acc.append(acc)

                unique_targets = np.unique(val_targets)
                num_classes_ = len(unique_targets)
                kappa = CohenKappa(task="multiclass", num_classes=num_classes_)(
                    torch.tensor(val_preds), torch.tensor(val_targets))
                val_kappa.append(kappa.item())

                f1 = F1Score(task="multiclass", num_classes=num_classes_)(
                    torch.tensor(val_preds), torch.tensor(val_targets))
                val_f1.append(f1.item())
                print(f"Validation Epoch {epoch+1}/{num_epochs}, Loss: {avg_val_loss:.4f}, Acc: {acc:.4f}, Kappa: {kappa.item():.4f}, F1: {f1.item():.4f}")
                
                # Early stopping check
                if acc > best_val_acc:
                    best_val_acc = acc
                    no_improvement_epochs = 0
                    torch.save(student_model.state_dict(), "best_student_model.pth")
                    print("New best model saved!")
                else:
                    no_improvement_epochs += 1
                    print(f"No improvement for {no_improvement_epochs}/150 epochs")
                
                if no_improvement_epochs >= 150:
                    print("Early stopping triggered after 150 epochs without improvement")
                    break

        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Kappa: {kappa.item():.4f}, F1: {f1.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log metrics
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"{timestamp},{epoch+1},{avg_loss:.4f},{acc:.4f},{kappa.item():.4f},{f1.item():.4f},{avg_val_loss:.4f},{val_acc[-1]:.4f},{val_kappa[-1]:.4f},{val_f1[-1]:.4f},{scheduler.get_last_lr()[0]:.6f},{no_improvement_epochs}"
        with open('training_log.txt', 'a') as log_file:
            log_file.write(log_entry + '\n')
        
        # Early stopping check
        if no_improvement_epochs >= 150:
            break
            
    return student_model

def create_plots():
    plt.figure(figsize=(10, 6))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Kappa
    plt.subplot(2, 2, 3)
    plt.plot(train_kappa, label='Training Kappa')
    plt.plot(val_kappa, label='Validation Kappa')
    plt.title('Cohen Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(train_f1, label='Training F1')
    plt.plot(val_f1, label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def save_plots():
    # Create a dictionary to hold all the data
    metrics = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_kappa': train_kappa,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_kappa': val_kappa,
        'val_f1': val_f1
    }
    
    # Create figure objects
    fig = plt.figure(figsize=(10, 6))
    create_plots()
    
    # Save the figure and metrics to a pickle file in current directory
    output_path = "/users/srenlund/training_metrics_normal_usleep.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(metrics, f)
        pickle.dump(fig, f)
    fig.savefig("training_metrics_plot_normal_usleep.png")
    print("Training metrics and plot saved to", output_path)
    print("Plot image saved as training_metrics_plot_normal_usleep.png")

def main():
    # Configuration
    batch_size = 128
    hdf5_file_path = "/users/srenlund/mnt/søren_data/shhs_spectro.hdf5"
    teacher_weights_path = (
        "/users/srenlund/ml_architectures/ml_architectures/usleep/weights/Depth10_CF05.ckpt"
    )
    student_weights_path = "/users/srenlund/student_model_normal_usleep.pth"
    
    # Setup data loading
    base_hdf5_path = os.path.dirname(hdf5_file_path)
    split = Split.random(base_hdf5_path=base_hdf5_path,
                        split_name="train",
                        split_percentages=(0.7, 0.2, 0.1))
    
    # Setup sampler
    train_sampler = Random_Sampler(split, num_epochs=1, num_iterations=10000)
    samplers = SamplerConfiguration(train_sampler, None, None)
    factory = Dataloader_Factory(training_batch_size=batch_size, samplers=samplers)
    data_loader = factory.training_loader(num_workers=4)
    
    # Initialize teacher model and load weights
    teacher_model = USleep(num_channels=2,
                         initial_filters=5,
                         complexity_factor=0.5,
                         depth=10)
    teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location='cpu'))
    teacher_model.eval()
    
    # Initialize student model
    student_model = USleep(num_channels = 2,
                        initial_filters = 3,
                        complexity_factor = 0.2,
                        progression_factor = 2,
                        depth = 10)
    # Initialize weights with Xavier initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    student_model.apply(init_weights)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
    
    # Train the student model
    train_student_model(teacher_model, student_model, data_loader, device, optimizer, scheduler)
    
    # Save the final model
    torch.save(student_model.state_dict(), "student_model_normal_usleep.pth")
    
    # Create and save plots
    create_plots()
    save_plots()
    
    # Save metrics to separate pickle file
    metrics = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_kappa': train_kappa,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_kappa': val_kappa,
        'val_f1': val_f1
    }
    with open('training_metrics_lists.pkl', 'wb') as f:
        pickle.dump(metrics, f)

if __name__ == "__main__":
    main()

