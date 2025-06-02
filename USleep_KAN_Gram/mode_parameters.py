import torch
import numpy as np
import matplotlib.pyplot as plt
from ml_architectures.ml_architectures.usleep.usleep import USleep
from Usleep_KAN_Gram_BN import USleep_BottleneckGRAM

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Initialize teacher model with the same configuration as in Gram_Sleep_Train.py
    teacher_model = USleep(num_channels=2,
                         initial_filters=5,
                         complexity_factor=0.5,
                         depth=10)
    
    # Initialize student model with the same configuration as in Gram_Sleep_Train.py
    student_model = USleep_BottleneckGRAM(num_channels=2,
                                        initial_filters=5,
                                        complexity_factor=0.99,
                                        progression_factor=1.4,
                                        num_classes=5,
                                        depth=10)
    
    # Count parameters
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    
    # Calculate reduction percentage
    reduction = (1 - (student_params / teacher_params)) * 100
    
    # Print results
    print(f"\nModel Parameter Comparison:")
    print(f"{'-'*40}")
    print(f"Teacher model (USleep):")
    print(f"  - Parameters: {teacher_params:,}")
    print(f"\nStudent model (USleep_BottleneckGRAM):")
    print(f"  - Parameters: {student_params:,}")
    print(f"\nParameter reduction: {reduction:.2f}%")
    print(f"{'-'*40}")
    
    # Create a bar chart for visualization
    models = ['Teacher (USleep)', 'Student (USleep_BottleneckGRAM)']
    params = [teacher_params, student_params]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, params, color=['#3498db', '#2ecc71'])
    plt.title('Model Parameter Comparison', fontsize=16)
    plt.ylabel('Number of Parameters', fontsize=14)
    plt.yscale('log')  # Log scale for better visualization if there's a big difference
    
    # Add parameter count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,}',
                ha='center', va='bottom', fontsize=12)
    
    # Add reduction percentage text
    plt.figtext(0.5, 0.01, f'Parameter reduction: {reduction:.2f}%', 
                ha='center', fontsize=14, bbox={"facecolor":"#f9f9f9", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the text
    plt.savefig('model_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
