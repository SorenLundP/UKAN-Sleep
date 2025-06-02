# UKAN-Sleep: Kolmogorov-Arnold Networks for Automated Sleep Staging

## Bachelor Thesis Project

---

## Abstract

This thesis investigates the integration of Kolmogorov-Arnold Networks (KANs) into automated sleep staging through the development of UKAN-Sleep, a parameter-efficient architecture that replaces standard convolutional layers with Bottleneck GRAM KAN layers in the established USleep framework. Sleep staging represents a critical bottleneck in sleep medicine, where manual polysomnography (PSG) scoring is labor-intensive and subject to inter-rater variability, while existing deep learning solutions face computational constraints limiting deployment in resource-constrained environments.

The proposed methodology integrates Bottleneck GRAM KAN convolutional layers, which employ learnable polynomial activation functions on network edges rather than fixed activations at nodes. Multiple model variants were implemented and evaluated using Knowledge Distillation and direct training approaches across three independent datasets (MASS-C1, ISRUC-SG1, and SVUH) using Cohen's Kappa as the primary performance metric.

Key findings demonstrate that KAN-inspired architectures consistently outperform conventional parameter reduction strategies. The UKAN-Sleep 10% (Student) model achieved κ = 0.569 ± 0.126 with approximately 90% parameter reduction (23k vs. 238k parameters), compared to κ = 0.541 ± 0.122 for conventional compression approaches. Directly trained variants showed stable performance across massive parameter reductions, indicating a performance plateau where substantial computational savings produce minimal performance degradation.

The investigation reveals important trade-offs between training paradigms: Knowledge Distillation achieves superior mean performance but exhibits increased cross-dataset variance, potentially limiting deployment reliability. The findings establish KAN-inspired approaches as viable pathways for creating computationally efficient sleep staging models while highlighting the importance of equivalent training conditions for fair architectural comparisons in biomedical contexts.

---

## Project Structure

- `USleep_KAN_Gram/` : KAN-based model implementations
- `CSDP-demo/` : Demos, validation scripts, and data loading
- `Common_Sleep_Data_Pipeline/` : Data preprocessing and dataset handling

## Requirements

See `requirements.txt` for dependencies. Install with:

```bash
pip install -r requirements.txt
```

## Usage

Example training/validation scripts are provided in `CSDP-demo/demos/`.

## Citation

If you use this work, please cite the thesis or contact the author for details.

---

## Contact

For questions or collaborations, please contact the thesis author.
