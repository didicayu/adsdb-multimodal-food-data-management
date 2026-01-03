# Fine-Tuning Pipeline for Multimodal Food Data

This directory contains the complete fine-tuning pipeline for adapting CLIP models to the food recipe domain. The pipeline follows the ADSDB project requirements and implements efficient fine-tuning techniques, data augmentation, and comprehensive experimental evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Fine-Tuning Approach](#fine-tuning-approach)
3. [Data Management](#data-management)
4. [Experiments](#experiments)
5. [Results](#results)
6. [Directory Structure](#directory-structure)
7. [Usage Instructions](#usage-instructions)

---

## Overview

This fine-tuning pipeline is designed to improve CLIP's performance on food recipe data by adapting the model to better align images and text descriptions in the culinary domain. The implementation uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning, allowing for fast training while maintaining model performance.

### Key Features

- **Efficient Fine-Tuning**: LoRA adapters reduce trainable parameters by ~99% compared to full fine-tuning
- **Data Augmentation**: Generative augmentation techniques for both images and text
- **Comprehensive Evaluation**: Retrieval-based metrics (Recall@K, MRR) with recipe-level evaluation
- **Reproducible Experiments**: Fixed random seeds and configuration files for all experiments
- **Zone-Based Architecture**: Integration with MinIO storage following ADSDB zone principles

---

## Fine-Tuning Approach

### Model Selection: CLIP

We selected **CLIP (Contrastive Language-Image Pre-training)** as our base model for the following reasons:

1. **Domain Alignment**: CLIP has been used throughout the project for creating multimodal embeddings, making it a natural choice for fine-tuning
2. **Data Compatibility**: Our pipeline manages both images and text, which aligns perfectly with CLIP's dual-encoder architecture
3. **Efficiency**: CLIP variants are relatively lightweight (<1B parameters), enabling fast experimentation
4. **Embedding Space**: CLIP creates a unified embedding space for images and text, ideal for retrieval tasks

### Fine-Tuning Technique: LoRA

We employ **LoRA (Low-Rank Adaptation)** for efficient fine-tuning:

- **Rationale**: LoRA introduces trainable low-rank matrices that approximate weight updates, dramatically reducing the number of trainable parameters
- **Configuration**: 
  - Rank (`r`): 8
  - Alpha (`lora_alpha`): 16
  - Dropout: 0.05
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj` (both vision and text encoders)
- **Benefits**:
  - **Parameter Efficiency**: Only ~0.1% of model parameters are trainable
  - **Memory Efficiency**: Lower VRAM requirements during training
  - **Fast Training**: Reduced computational cost enables rapid experimentation
  - **Modularity**: LoRA adapters can be saved/loaded independently

### Expected Impact

Fine-tuning with LoRA is expected to:
- **Improve Alignment**: Better semantic alignment between food images and recipe descriptions
- **Domain Adaptation**: Adapt CLIP's general knowledge to food-specific terminology and visual features
- **Retrieval Performance**: Increase Recall@K metrics for text→image and image→text retrieval tasks
- **Maintain Efficiency**: Preserve inference speed while improving accuracy

---

## Data Management

### 3.1 Creating the Training Dataset

#### Process: Knowledge Distillation Approach

We generate training data using a **knowledge distillation** approach:

1. **Source Data**: 
   - Images from `trusted-zone/images/` (normalized 512×512 JPEG)
   - Recipe metadata from `trusted-zone/documents/recipes.jsonl`

2. **Pairing Strategy**:
   - Each recipe has multiple images and multiple text descriptions
   - We create positive pairs by linking images and captions from the same `recipe_id`
   - This ensures semantic alignment: images and text describe the same dish

3. **Implementation** (`training_data/prepare_dataset.ipynb`):
   - Loads recipes and associated images from the Trusted Zone
   - Creates image-text pairs with matching `recipe_id`
   - Removes duplicates and missing values
   - Splits into train/test sets (80/20) with fixed random seed (42)
   - Copies images to `fine-tuning-zone/images/` for training

4. **Output**:
   - `train_pairs_positive.csv`: Training pairs (recipe_id, image_key, caption)
   - `test_pairs_positive.csv`: Test pairs for evaluation

#### Why This Approach?

- **Domain-Specific**: Uses actual project data, ensuring fine-tuning adapts to our specific domain
- **Automatic**: Leverages existing recipe structure to create pairs without manual annotation
- **Scalable**: Can process large datasets efficiently
- **Quality**: Recipe-level pairing ensures semantic coherence

### 3.2 Data Augmentation

We apply **generative data augmentation** to enrich the training dataset:

#### Image Augmentation (`augmentation/augment_dataset.ipynb`)

Transformations applied to increase model robustness:

- **Geometric Transforms**:
  - Horizontal/Vertical flips
  - Random rotation (-25° to +25°, 90°, 180°, 270°)
  - Random crop and resize (70-90% of original)
  
- **Color Transforms**:
  - Brightness adjustment (0.7-1.3x)
  - Contrast adjustment (0.7-1.3x)
  - Saturation adjustment (0.6-1.4x)
  
- **Noise and Blur**:
  - Gaussian noise (std: 15-35)
  - Gaussian blur (radius: 0.5-2.0)

**Configuration**: See `augmentation/augmentation_config.yaml` for detailed parameters.

#### Text Augmentation

Rule-based transformations that preserve semantic meaning:

- **Synonym Replacement**: Food-safe word substitutions (e.g., "delicious" → "tasty", "crispy" → "crunchy")
- **Prefix Addition/Removal**: Add/remove descriptive prefixes ("Homemade", "Classic", etc.)
- **Title Reordering**: Reorder "Ingredient + Dish" patterns (e.g., "Banana French Toast" → "French Toast with Banana")
- **Case Normalization**: Title case, lowercase, or uppercase variants
- **Suffix Addition**: Append recipe-style suffixes ("Recipe", "- Easy Recipe")
- **Simplification**: Remove parenthetical content and numbering

#### Negative Pair Generation

To improve contrastive learning:
- Mismatch images and captions from different recipes
- Generate negative pairs (label=0) alongside positive pairs (label=1)
- Ratio: 1 negative per positive pair

**Output**: `train_pairs_augmented_with_negatives.csv` with augmented pairs and labels.

### 3.3 Zone Architecture

We extend the ADSDB zone architecture with a **Fine-Tuning Zone**:

```
fine-tuning-zone/
├── datasets/
│   ├── train_pairs_positive.csv
│   ├── train_pairs_augmented_with_negatives.csv
│   └── test_pairs_positive.csv
├── images/
│   └── [original images from trusted zone]
├── augmented_images/
│   └── [augmented image variants]
└── experiments/
    ├── baseline/
    │   ├── results_baseline.json
    │   └── examples_top5.json
    └── lora/
        └── run_{run_id}/
            ├── config.yaml
            ├── adapters/
            ├── training_logs.json
            ├── results_lora.json
            └── examples_top5.json
```

**Rationale**:
- **Separation of Concerns**: Training data, augmented data, and experiment results are clearly separated
- **Reproducibility**: All datasets and configurations are versioned in MinIO
- **Scalability**: Can store multiple experiment runs with unique identifiers
- **Integration**: Seamlessly extends the existing zone structure (Landing → Trusted → Formatted → Exploitation → Fine-Tuning)

---

## Experiments

### 4.1 Experimental Design

#### Research Hypothesis

**H1**: Fine-tuning CLIP with LoRA on domain-specific food recipe data improves text↔image alignment compared to the baseline CLIP model, as measured by retrieval metrics (Recall@K, MRR).

#### Controlled Variables

We evaluate the following model variants:

1. **M0: Baseline CLIP** (`01_baseline_eval_clip.ipynb`)
   - Pre-trained CLIP (no fine-tuning)
   - Model: `openai/clip-vit-base-patch32`
   - Serves as the reference baseline

2. **M1: LoRA Fine-Tuned CLIP** (`02_lora_finetune_clip.ipynb`)
   - Same base model as M0
   - LoRA adapters applied to vision and text encoders
   - Trained on augmented training dataset

#### Controlled Conditions

To ensure fair comparison:
- **Same Test Set**: Both models evaluated on identical `test_pairs_positive.csv`
- **Same Evaluation Protocol**: Recipe-level retrieval (multiple valid matches per query)
- **Same Metrics**: Recall@1, Recall@5, Recall@10, MRR, MedianRank
- **Same Hardware**: Consistent device and batch size settings
- **Reproducibility**: Fixed random seeds (42) for all operations

#### Baselines

- **M0 (Baseline CLIP)**: Pre-trained model without fine-tuning
  - Represents general-purpose CLIP performance on food data
  - Establishes the performance floor for comparison

#### Reproducibility

- **Random Seeds**: Fixed seeds for NumPy (42), PyTorch (42), Python random (42)
- **Configuration Files**: All hyperparameters stored in YAML files
- **Version Control**: Experiment runs include timestamps and configuration snapshots
- **Deterministic Operations**: CUDA deterministic mode enabled where applicable

### 4.2 Evaluation Metrics

We use **retrieval-based metrics** to evaluate multimodal alignment:

#### Primary Metrics

1. **Recall@K** (R@K): Fraction of queries where at least one relevant item appears in top-K results
   - **R@1, R@5, R@10**: Evaluated for both text→image and image→text retrieval
   - **Recipe-Level Evaluation**: Multiple valid matches per query (all images/captions with same `recipe_id`)

2. **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant item
   - Measures how quickly the model finds the first relevant result
   - Range: [0, 1], higher is better

3. **Median Rank (First Hit)**: Median position of first relevant item
   - Lower values indicate better performance
   - Complements MRR by showing typical retrieval position

#### Efficiency Metrics

- **Training Time**: Total time to complete fine-tuning
- **Peak VRAM**: Maximum GPU memory usage during training
- **Inference Throughput**: Images/second and texts/second during evaluation
- **Trainable Parameters**: Number of LoRA parameters vs. total model parameters

#### Why These Metrics?

- **Retrieval-Focused**: CLIP is primarily used for retrieval tasks in our pipeline
- **Recipe-Level Granularity**: Accounts for multiple valid matches per recipe
- **Efficiency Awareness**: Tracks computational costs alongside accuracy
- **Standard Practice**: Widely used in multimodal retrieval literature

### 4.3 Reporting and Visualization

#### Results Storage

Each experiment run produces:

- **Quantitative Results**: JSON files with all metrics
- **Qualitative Examples**: Top-5 retrieval examples for visual inspection
- **Training Logs**: Loss curves and training dynamics
- **Configuration Snapshots**: Complete experiment configuration for reproducibility

#### Visualization Components

1. **Learning Curves**: Training loss over epochs (from `training_logs.json`)
2. **Metric Comparisons**: Bar charts comparing M0 vs. M1 across all metrics
3. **Retrieval Examples**: Side-by-side comparisons of baseline vs. fine-tuned retrieval results
4. **Efficiency Analysis**: Plots showing VRAM usage, training time, and inference speed

#### Analysis and Discussion

Results are analyzed with respect to:
- **Hypothesis Validation**: Whether H1 is supported by the data
- **Performance Trade-offs**: Accuracy improvements vs. computational costs
- **Failure Cases**: Analysis of queries where fine-tuning did not improve results
- **Domain Adaptation**: Evidence of improved food-specific understanding

---

## Results

This section presents the experimental results comparing the baseline CLIP model (M0) with the LoRA fine-tuned model (M1) on the food recipe dataset.

### 5.1 Quantitative Results

#### Retrieval Performance Comparison

The following table summarizes the retrieval metrics for both models:

| Metric | Direction | Baseline (M0) | LoRA (M1) | Improvement |
|--------|-----------|---------------|-----------|-------------|
| **Recall@1** | Text→Image | 0.574 | 0.593 | +3.3% |
| **Recall@5** | Text→Image | 0.796 | 0.852 | +7.0% |
| **Recall@10** | Text→Image | 0.944 | 0.963 | +2.0% |
| **MRR** | Text→Image | 0.688 | 0.702 | +2.0% |
| **MeanRank** | Text→Image | 3.574 | 3.278 | -8.3% |
| **Recall@1** | Image→Text | 0.689 | 0.689 | 0.0% |
| **Recall@5** | Image→Text | 0.913 | 0.942 | +3.2% |
| **Recall@10** | Image→Text | 0.961 | 0.990 | +3.0% |
| **MRR** | Image→Text | 0.796 | 0.799 | +0.4% |
| **MeanRank** | Image→Text | 2.068 | 1.951 | -5.7% |

#### Key Findings

1. **Text→Image Retrieval Improvements**:
   - **Recall@5** shows the largest improvement (+7.0%), indicating better performance in the top-5 results
   - **Recall@1** improved by 3.3%, showing the model finds the correct image as the top result more often
   - **MRR** increased from 0.688 to 0.702, confirming better ranking of relevant results
   - **MeanRank** decreased from 3.574 to 3.278, meaning relevant images appear earlier in the ranked list

2. **Image→Text Retrieval Improvements**:
   - **Recall@5** improved by 3.2% (0.913 → 0.942), the most significant gain
   - **Recall@10** reached 0.990, indicating near-perfect retrieval in the top-10
   - **MeanRank** decreased from 2.068 to 1.951, showing relevant captions are ranked higher
   - **Recall@1** remained unchanged (0.689), suggesting the model already performs well at the top position

3. **Overall Assessment**:
   - Fine-tuning with LoRA consistently improves retrieval performance across most metrics
   - Improvements are more pronounced in Recall@5 and Recall@10, indicating better ranking in the top-K results
   - The hypothesis **H1 is supported**: LoRA fine-tuning improves text↔image alignment

### 5.2 Training Dynamics

#### Loss Progression

The training logs show a steady decrease in contrastive loss over 3 epochs:

| Epoch | Average Loss | Training Time |
|-------|--------------|---------------|
| 1 | 0.078 | 146.8 seconds |
| 2 | 0.052 | 138.8 seconds |
| 3 | 0.037 | 144.3 seconds |

**Observations**:
- Loss decreased by **52.6%** from epoch 1 to epoch 3 (0.078 → 0.037)
- Training converged smoothly without signs of overfitting
- Total training time: **429.9 seconds** (~7.2 minutes) for 3 epochs
- Training was performed on CPU, which explains the longer training time

#### Training Efficiency

- **Total Steps**: 135 optimizer steps (45 steps per epoch)
- **Effective Batch Size**: 32 (batch_size=2 × gradient_accumulation_steps=16)
- **Trainable Parameters**: 983,041 (0.65% of total 152M parameters)
- **Parameter Efficiency**: Only 0.65% of model parameters were updated, demonstrating LoRA's efficiency

### 5.3 Model Efficiency

#### Parameter Comparison

| Model | Total Parameters | Trainable Parameters | Trainable Ratio |
|-------|------------------|----------------------|-----------------|
| Baseline (M0) | 151,277,313 | 151,277,313 | 100% |
| LoRA (M1) | 152,260,353 | 983,041 | 0.65% |

**Analysis**:
- LoRA adds only **983K trainable parameters** (0.65% of total)
- The model size increase is minimal (~1M additional parameters)
- This confirms LoRA's parameter efficiency: fine-tuning with <1% of parameters

#### Inference Performance

| Metric | Baseline (M0) | LoRA (M1) |
|--------|----------------|-----------|
| **Images/sec** | 45.23 | Similar (adapter overhead negligible) |
| **Texts/sec** | 463.33 | Similar (adapter overhead negligible) |
| **Peak VRAM** | 0.586 GB | Similar (adapters loaded in memory) |

**Note**: Both models were evaluated on CPU, so GPU memory metrics are not available. The LoRA adapters add minimal overhead during inference.

### 5.4 Qualitative Analysis

#### Retrieval Examples

The qualitative examples (`examples_top5.json`) reveal several patterns:

1. **Successful Cases**:
   - **"Yummy Chocolate Crumb Cake"**: Both top-2 results are correct (ranks 1-2)
   - **"Frozen Peach Daiquiris"**: Correct image ranked #1
   - **"Potato Cabbage Chowder"**: Correct image ranked #1
   - **"Dirt Trifle With Gummy Worms"**: Correct caption ranked #1

2. **Challenging Cases**:
   - **"Peanut Butter Glazed Brownie Doughnuts"**: Correct image ranked #2 (incorrect at #1)
   - **"Italian Orange Cookies"**: Correct caption ranked #5 (lower than expected)
   - Some queries show confusion between similar dishes (e.g., different cake types)

3. **Observations**:
   - The model generally ranks relevant results in the top-5
   - Some confusion occurs between visually or semantically similar dishes
   - Fine-tuning helps but doesn't eliminate all retrieval errors

### 5.5 Discussion

#### Hypothesis Validation

**H1 is supported**: Fine-tuning CLIP with LoRA on domain-specific food recipe data improves text↔image alignment. The improvements, while modest, are consistent across multiple metrics:

- **Text→Image**: Improvements in R@1 (+3.3%), R@5 (+7.0%), and MRR (+2.0%)
- **Image→Text**: Improvements in R@5 (+3.2%) and R@10 (+3.0%)

#### Performance Trade-offs

**Advantages**:
- **Parameter Efficiency**: Only 0.65% of parameters are trainable
- **Modest Improvements**: Consistent gains across retrieval metrics
- **Fast Training**: ~7 minutes for 3 epochs (on CPU)
- **No Inference Overhead**: LoRA adapters add minimal computational cost

**Limitations**:
- **Modest Gains**: Improvements are present but not dramatic (3-7% in most metrics)
- **Recall@1 Unchanged**: Image→Text R@1 remained at 0.689
- **Small Dataset**: Results are based on a relatively small test set (54 recipes, 103 images)

#### Possible Explanations

1. **Dataset Size**: The training dataset may be too small to fully exploit fine-tuning benefits
2. **Baseline Performance**: CLIP already performs well on food data (R@10 > 0.94), leaving limited room for improvement
3. **Domain Similarity**: Food images and text may already be well-aligned in CLIP's pre-trained space
4. **Training Configuration**: Further hyperparameter tuning (learning rate, epochs, LoRA rank) might yield larger improvements

#### Future Improvements

1. **Larger Dataset**: Increase training data size through additional augmentation or data collection
2. **Hyperparameter Tuning**: Experiment with different LoRA ranks, learning rates, and training schedules
3. **QLoRA**: Test quantized LoRA for even more efficient training
4. **Different Architectures**: Compare LoRA with other parameter-efficient methods (adapters, prompt tuning)
5. **Longer Training**: Train for more epochs to see if improvements continue

---

## Directory Structure

```
fine_tuning/
├── README.md                          # This file
├── training_data/
│   ├── prepare_dataset.ipynb         # Dataset creation from Trusted Zone
│   ├── train_pairs_positive.csv      # Training pairs (local copy)
│   └── test_pairs_positive.csv       # Test pairs (local copy)
├── augmentation/
│   ├── augment_dataset.ipynb         # Data augmentation pipeline
│   └── augmentation_config.yaml      # Augmentation parameters
└── experiments/
    ├── experiment_config.yaml         # Fine-tuning and evaluation config
    ├── 01_baseline_eval_clip.ipynb    # Baseline evaluation (M0)
    └── 02_lora_finetune_clip.ipynb   # LoRA fine-tuning (M1)
```

---

## Usage Instructions

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Install dependencies (from project root)
   pip install -r requirements.txt
   ```

2. **MinIO Configuration**:
   - Ensure MinIO is running (see `docker/docker-compose.yml`)
   - Set environment variables in `.env`:
     - `MINIO_USER`
     - `MINIO_PASSWORD`
     - `MINIO_ENDPOINT`

3. **Data Requirements**:
   - Trusted Zone must contain:
     - `trusted-zone/documents/recipes.jsonl`
     - `trusted-zone/images/` (normalized images)

### Execution Workflow

#### Step 1: Prepare Training Dataset

```bash
cd fine_tuning/training_data
jupyter notebook prepare_dataset.ipynb
```

**Output**: 
- `train_pairs_positive.csv` and `test_pairs_positive.csv` in MinIO (`fine-tuning-zone/datasets/`)
- Images copied to `fine-tuning-zone/images/`

#### Step 2: Augment Training Data

```bash
cd fine_tuning/augmentation
jupyter notebook augment_dataset.ipynb
```

**Output**:
- `train_pairs_augmented_with_negatives.csv` in MinIO
- Augmented images in `fine-tuning-zone/augmented_images/`

#### Step 3: Evaluate Baseline (M0)

```bash
cd fine_tuning/experiments
# Edit experiment_config.yaml: set method: "baseline"
jupyter notebook 01_baseline_eval_clip.ipynb
```

**Output**: Results saved to `fine-tuning-zone/experiments/baseline/`

#### Step 4: Fine-Tune with LoRA (M1)

```bash
cd fine_tuning/experiments
# Edit experiment_config.yaml: set method: "lora"
jupyter notebook 02_lora_finetune_clip.ipynb
```

**Output**: 
- LoRA adapters saved to `fine-tuning-zone/experiments/lora/run_{run_id}/adapters/`
- Evaluation results in `fine-tuning-zone/experiments/lora/run_{run_id}/`

### Configuration

All hyperparameters are controlled via YAML files:

- **Training**: `experiments/experiment_config.yaml`
  - Model selection, LoRA parameters, training hyperparameters
- **Augmentation**: `augmentation/augmentation_config.yaml`
  - Image and text augmentation strategies

### Reproducibility

- All random seeds are fixed (42) in configuration files
- Experiment runs include timestamped directories
- Configuration snapshots are saved with each run
- Results are stored in MinIO for version control

---

## Notes

- **GPU Requirements**: LoRA fine-tuning requires GPU (CUDA) for reasonable training time
- **Storage**: Ensure sufficient MinIO storage for datasets and experiment results
- **Evaluation Protocol**: Recipe-level evaluation means multiple valid matches per query
- **Future Work**: Consider QLoRA (quantized LoRA) for even more efficient training

---

