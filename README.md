# ADSDB Multimodal Food Data Management Pipeline

This repository contains a comprehensive multimodal data pipeline for food recipe management. The system transforms raw data from external sources through multiple processing zones, ultimately creating a searchable knowledge base of recipes with images using ChromaDB.

## 🚀 Quick Start

### All Users

Run the pipeline with these commands:

```bash
# Using CLI (recommended)
python app/cli.py run

# Or using orchestrator directly (always works)
python app/orchestrate.py
```

### 🪟 Windows Users

**Important**: Windows PowerShell/CMD may have issues with emoji display in the CLI. Use these commands:

```bash
# Set UTF-8 encoding for proper display
chcp 65001

# Then run CLI commands normally
python app/cli.py run
```

## 🏗️ Architecture Overview

The pipeline follows a data lake architecture with multiple zones organized in two main parts:

### Part 1: Data Pipeline (Zones 1-4)

1. **Landing Zone**: Raw data ingestion and initial organization
2. **Formatted Zone**: Data cleaning, joining, and standardization
3. **Trusted Zone**: Quality-controlled, filtered data ready for analysis
4. **Exploitation Zone**: Vector embeddings and search capabilities (documents + images)

### Part 2: Fine-Tuning (Zone 5)

5. **Fine-Tuning Zone**: Model fine-tuning with LoRA, data augmentation, and experimental evaluation

## 📁 Project Structure

```
adsdb-multimodal-food-data-management/
├── app/                           # Main application code
│   ├── pipeline.yaml             # Centralized pipeline configuration
│   ├── requirements.txt           # Python dependencies
│   ├── cli.py                    # Command-line interface
│   ├── orchestrate.py            # Main pipeline orchestrator
│   ├── utils/                    # Shared utilities
│   │   ├── __init__.py           # Utility exports
│   │   ├── config.py             # Configuration management
│   │   ├── monitoring.py         # Performance monitoring
│   │   └── shared.py             # Shared utilities (S3Client, Logger, etc.)
│   └── zones/                    # Processing zones
│       ├── landing_zone/          # Raw data ingestion
│       │   ├── temporal_landing.py
│       │   └── persistent_landing.py
│       ├── formatted_zone/        # Data cleaning and joining
│       │   ├── formatted_documents.py
│       │   └── formatted_images.py
│       ├── trusted_zone/          # Quality control
│       │   ├── trusted_documents.py
│       │   └── trusted_images.py
│       ├── exploitation_zone/     # Vector embeddings
│       │   ├── exploitation_documents.py
│       │   └── exploitation_images.py
│       └── multimodal_tasks/      # Multimodal task implementations
│           ├── task1_retrieval.py
│           ├── task2.py
│           └── task3_rag.py
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── landing_zone/
│   ├── formatted_zone/
│   ├── trusted_zone/
│   └── exploitation_zone/
├── fine_tuning/                   # Fine-tuning pipeline (Part 2)
│   ├── training_data/             # Dataset preparation
│   │   ├── prepare_dataset.ipynb
│   │   ├── train_pairs_positive.csv
│   │   └── test_pairs_positive.csv
│   ├── augmentation/              # Data augmentation
│   │   ├── augment_dataset.ipynb
│   │   └── augmentation_config.yaml
│   └── experiments/               # Fine-tuning experiments
│       ├── experiment_config.yaml
│       ├── 01_baseline_eval_clip.ipynb
│       └── 02_lora_finetune_clip.ipynb
├── docker/                       # Docker configuration
│   ├── docker-compose.yml
│   ├── init-minio.sh
│   └── _minio_data/             # MinIO data directory
├── tests/                        # Test suite
│   ├── unit/
│   └── integration/
├── app/zones/landing_zone/       # Index files
│   ├── image_index.json
│   └── recipes_index.json
├── app/zones/trusted_zone/       # Recipe IDs with images
│   └── recipe_ids_with_images.json
├── app/zones/exploitation_zone/   # ChromaDB data
│   ├── chroma_documents/
│   ├── chroma_images/
│   └── chroma_exploitation/       # Unified multimodal collection
├── app/zones/multimodal_tasks/    # Multimodal task implementations
│   ├── task1_retrieval.py         # Text-to-text and image-to-image search
│   ├── task2.py                   # Unified multimodal search
│   ├── task3_rag.py              # RAG with LLaVA
│   └── README.md
└── README.md
```

## 🚀 Quick Start

### Prerequisites

-   **Python 3.8+** with pip
-   **MinIO server** running locally or remotely
-   **Hugging Face account** (for dataset access)
-   **Git** for version control

### Installation

1. **Clone the repository**

    ```bash
    git clone <repository-url>
    cd adsdb-multimodal-food-data-management
    ```

2. **Install Python dependencies**

    ```bash
    pip install -r app/requirements.txt
    ```

3. **Configure environment variables**

    ```bash
    cp app/env.example .env
    # Edit .env with your actual values
    ```

4. **Initialize the pipeline**

    ```bash
    python app/cli.py init
    ```

5. **Validate configuration**
    ```bash
    python app/cli.py validate
    ```

### Environment Variables (.env)

```bash
# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here
HF_ORGA=your_organization_name
HF_DATASET=your_dataset_name
HF_REV=main

# MinIO Configuration
MINIO_USER=minio
MINIO_PASSWORD=minio12345
MINIO_ENDPOINT=http://localhost:9000

```

## 🔄 Pipeline Stages

### 1. Temporal Landing (`temporal_landing`)

-   **Purpose**: Ingest raw data from Hugging Face datasets
-   **Input**: External Hugging Face dataset
-   **Output**: Raw files in MinIO landing zone
-   **Key Features**:
    -   Handles various file formats (JSON, JSONL, images)
    -   Processes layer2.json for image URLs
    -   Applies naming conventions with hashing
    -   Creates image and recipe indexes

### 2. Persistent Landing (`persistent_landing`)

-   **Purpose**: Organize raw data by type and apply naming conventions
-   **Input**: Raw files from temporal landing
-   **Output**: Organized files in persistent landing structure
-   **Key Features**:
    -   File type detection (images vs documents)
    -   Standardized naming: `type$dataset$timestamp$name.ext`
    -   Separates images and documents

### 3. Formatted Documents (`formatted_documents`)

-   **Purpose**: Join all recipes and remove irrelevant data
-   **Input**: Organized documents from persistent landing
-   **Output**: Single JSONL file with joined recipes
-   **Key Features**:
    -   Merges recipes by ID with tagged field collisions
    -   Removes specified fields (url, partition)
    -   Handles multiple JSON formats (JSON, JSONL, arrays)

### 4. Formatted Images (`formatted_images`)

-   **Purpose**: Process and organize images
-   **Input**: Organized images from persistent landing
-   **Output**: Processed images in formatted zone
-   **Key Features**:
    -   Quality screening (size, aspect ratio, blur detection)
    -   Per-recipe deduplication using perceptual hashing
    -   Image normalization to standard format

### 5. Trusted Images (`trusted_images`)

-   **Purpose**: Extract recipe IDs and copy filtered images
-   **Input**: Processed images from formatted zone
-   **Output**: Quality-controlled images in trusted zone
-   **Key Features**:
    -   Recipe ID extraction from filenames
    -   Quality filtering and deduplication
    -   Generates recipe IDs file for document processing
    -   Image normalization to 512x512 JPEG

### 6. Trusted Documents (`trusted_documents`)

-   **Purpose**: Filter documents to keep only those with images
-   **Input**: Joined documents from formatted zone
-   **Output**: Filtered documents in trusted zone
-   **Key Features**:
    -   Filters by recipe IDs that have images
    -   Applies nutrition quality controls (IQR outlier detection)
    -   Text cleaning and normalization
    -   Removes duplicated nutrition data

### 7. Exploitation Documents (`exploitation_documents`)

-   **Purpose**: Generate embeddings and store in ChromaDB
-   **Input**: Filtered documents from trusted zone
-   **Output**: Vector embeddings in ChromaDB
-   **Key Features**:
    -   Uses SentenceTransformer for embeddings
    -   Batch processing for efficiency
    -   Metadata extraction (FSA nutrition lights)
    -   Semantic search capabilities

### 8. Exploitation Images (`exploitation_images`)

-   **Purpose**: Generate image embeddings and store in ChromaDB
-   **Input**: Quality-controlled images from trusted zone
-   **Output**: Vector embeddings in ChromaDB
-   **Key Features**:
    -   Uses OpenCLIP for image embeddings
    -   Cross-modal search capabilities
    -   Batch processing for efficiency
    -   Image metadata extraction

### 9. Task 1: Multimodal Retrieval (`task1_retrieval`)

-   **Purpose**: Perform multimodal retrieval operations on recipe data
-   **Input**: ChromaDB collections (documents and images)
-   **Output**: Similar recipes and images based on queries
-   **Key Features**:
    -   Text-to-text search (find similar recipes)
    -   Image-to-image search (find visually similar images)
    -   No external services required

### 10. Task 2: Multimodal Search (`task2_multimodal_search`)

-   **Purpose**: Unified multimodal search on combined collection
-   **Input**: ChromaDB multimodal collection (text + images)
-   **Output**: Cross-modal retrieval results
-   **Key Features**:
    -   Text queries return both recipes AND images
    -   Image queries return both images AND recipes
    -   Single unified collection for cross-modal search
    -   Distance statistics for match quality

### 11. Task 3: Multimodal RAG with LLaVA (`task3_rag`)

-   **Purpose**: Retrieval-augmented generation with vision and language
-   **Input**: ChromaDB collections + user queries
-   **Output**: Generated responses with retrieved context
-   **Key Features**:
    -   Text and image retrieval
    -   LLaVA model for multimodal generation via Ollama
    -   Contextual responses using retrieved recipes and images
    -   Requires Ollama server running

---

## 🎯 Part 2: Fine-Tuning Pipeline

This section covers the fine-tuning pipeline for adapting CLIP models to the food recipe domain. This is the second part of the ADSDB project, focusing on model adaptation and experimental evaluation.

### Overview

The fine-tuning pipeline improves CLIP's performance on food recipe data by adapting the model to better align images and text descriptions in the culinary domain. The implementation uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.

**Key Features**:

-   **Efficient Fine-Tuning**: LoRA adapters reduce trainable parameters by ~99% compared to full fine-tuning
-   **Data Augmentation**: Generative augmentation techniques for both images and text
-   **Comprehensive Evaluation**: Retrieval-based metrics (Recall@K, MRR) with recipe-level evaluation
-   **Reproducible Experiments**: Fixed random seeds and configuration files for all experiments
-   **Zone-Based Architecture**: Integration with MinIO storage following ADSDB zone principles

### Fine-Tuning Approach

#### Model Selection: CLIP

We selected **CLIP (Contrastive Language-Image Pre-training)** as our base model for the following reasons:

1. **Domain Alignment**: CLIP has been used throughout the project for creating multimodal embeddings, making it a natural choice for fine-tuning
2. **Data Compatibility**: Our pipeline manages both images and text, which aligns perfectly with CLIP's dual-encoder architecture
3. **Efficiency**: CLIP variants are relatively lightweight (<1B parameters), enabling fast experimentation
4. **Embedding Space**: CLIP creates a unified embedding space for images and text, ideal for retrieval tasks

#### Fine-Tuning Technique: LoRA

We employ **LoRA (Low-Rank Adaptation)** for efficient fine-tuning:

-   **Rationale**: LoRA introduces trainable low-rank matrices that approximate weight updates, dramatically reducing the number of trainable parameters
-   **Configuration**:
    -   Rank (`r`): 8
    -   Alpha (`lora_alpha`): 16
    -   Dropout: 0.05
    -   Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj` (both vision and text encoders)
-   **Benefits**:
    -   **Parameter Efficiency**: Only ~0.1% of model parameters are trainable
    -   **Memory Efficiency**: Lower VRAM requirements during training
    -   **Fast Training**: Reduced computational cost enables rapid experimentation
    -   **Modularity**: LoRA adapters can be saved/loaded independently

### Data Management for Fine-Tuning

#### Creating the Training Dataset

We generate training data using the following approach:

1. **Source Data**:

    - Images from `trusted-zone/images/` (normalized 512×512 JPEG)
    - Recipe metadata from `trusted-zone/documents/recipes.jsonl`

2. **Pairing Strategy**:

    - Each recipe has multiple images and multiple text descriptions
    - We create positive pairs by linking images and captions from the same `recipe_id`
    - This ensures semantic alignment: images and text describe the same dish

3. **Implementation** (`fine_tuning/training_data/prepare_dataset.ipynb`):

    - Loads recipes and associated images from the Trusted Zone
    - Creates image-text pairs with matching `recipe_id`
    - Removes duplicates and missing values
    - Splits into train/test sets (80/20) with fixed random seed (42)
    - Copies images to `fine-tuning-zone/images/` for training

4. **Output**:
    - `train_pairs_positive.csv`: Training pairs (recipe_id, image_key, caption)
    - `test_pairs_positive.csv`: Test pairs for evaluation

#### Data Augmentation

We apply **generative data augmentation** to enrich the training dataset:

**Image Augmentation** (`fine_tuning/augmentation/augment_dataset.ipynb`):

-   **Geometric Transforms**: Horizontal/Vertical flips, random rotation, crop and resize
-   **Color Transforms**: Brightness, contrast, and saturation adjustments
-   **Noise and Blur**: Gaussian noise and blur for robustness

**Text Augmentation**:

-   **Synonym Replacement**: Food-safe word substitutions
-   **Prefix Addition/Removal**: Add/remove descriptive prefixes
-   **Title Reordering**: Reorder "Ingredient + Dish" patterns
-   **Case Normalization**: Title case, lowercase, or uppercase variants
-   **Suffix Addition**: Append recipe-style suffixes
-   **Simplification**: Remove parenthetical content and numbering

**Negative Pair Generation**: Mismatch images and captions from different recipes to improve contrastive learning.

#### Fine-Tuning Zone Architecture

We extend the zones architecture with a **Fine-Tuning Zone**:

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

### Fine-Tuning Experiments

#### Experimental Design

**Research Hypothesis H1**: Fine-tuning CLIP with LoRA on domain-specific food recipe data improves text↔image alignment compared to the baseline CLIP model, as measured by retrieval metrics (Recall@K, MRR).

**Model Variants**:

1. **M0: Baseline CLIP** (`fine_tuning/experiments/01_baseline_eval_clip.ipynb`)

    - Pre-trained CLIP (no fine-tuning)
    - Model: `openai/clip-vit-base-patch32`
    - Serves as the reference baseline

2. **M1: LoRA Fine-Tuned CLIP** (`fine_tuning/experiments/02_lora_finetune_clip.ipynb`)
    - Same base model as M0
    - LoRA adapters applied to vision and text encoders
    - Trained on augmented training dataset

**Controlled Conditions**:

-   Same test set for both models
-   Same evaluation protocol (recipe-level retrieval)
-   Same metrics (Recall@1, Recall@5, Recall@10, MRR, MedianRank)
-   Fixed random seeds (42) for reproducibility

#### Evaluation Metrics

**Primary Metrics**:

-   **Recall@K** (R@K): Fraction of queries where at least one relevant item appears in top-K results
-   **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant item
-   **Median Rank**: Median position of first relevant item

**Efficiency Metrics**:

-   Training time, peak VRAM, inference throughput
-   Trainable parameters vs. total model parameters

### Running the Fine-Tuning Pipeline

#### Prerequisites

1. **Complete Part 1**: Ensure the Trusted Zone contains:

    - `trusted-zone/documents/recipes.jsonl`
    - `trusted-zone/images/` (normalized images)

2. **Install Dependencies**: All dependencies from `app/requirements.txt` plus fine-tuning specific packages

#### Execution Workflow

**Step 1: Prepare Training Dataset**

```bash
cd fine_tuning/training_data
jupyter notebook prepare_dataset.ipynb
```

**Output**: `train_pairs_positive.csv` and `test_pairs_positive.csv` in MinIO (`fine-tuning-zone/datasets/`)

**Step 2: Augment Training Data**

```bash
cd fine_tuning/augmentation
jupyter notebook augment_dataset.ipynb
```

**Output**: `train_pairs_augmented_with_negatives.csv` in MinIO

**Step 3: Evaluate Baseline (M0)**

```bash
cd fine_tuning/experiments
# Edit experiment_config.yaml: set method: "baseline"
jupyter notebook 01_baseline_eval_clip.ipynb
```

**Output**: Results saved to `fine-tuning-zone/experiments/baseline/`

**Step 4: Fine-Tune with LoRA (M1)**

```bash
cd fine_tuning/experiments
# Edit experiment_config.yaml: set method: "lora"
jupyter notebook 02_lora_finetune_clip.ipynb
```

**Output**: LoRA adapters and evaluation results in `fine-tuning-zone/experiments/lora/run_{run_id}/`

#### Configuration

All hyperparameters are controlled via YAML files:

-   **Training**: `fine_tuning/experiments/experiment_config.yaml`
-   **Augmentation**: `fine_tuning/augmentation/augmentation_config.yaml`

#### Reproducibility

-   All random seeds are fixed (42) in configuration files
-   Experiment runs include timestamped directories
-   Configuration snapshots are saved with each run
-   Results are stored in MinIO for version control

## 🖥️ Running the Pipeline

### All Users

The pipeline can be run using the CLI or orchestrator directly:

```bash
# Using CLI (recommended)
python app/cli.py run

# Or using orchestrator directly (always works)
python app/orchestrate.py
```

### Windows Users (PowerShell/CMD)

**Important**: Windows terminals may have issues with emoji display. Use these commands:

```bash
# Set UTF-8 encoding for proper display
chcp 65001

# Then run CLI commands
python app/cli.py run
```

### CLI Commands

#### Full Pipeline

```bash
# Using CLI (recommended)
python app/cli.py run

# Or using orchestrator directly (always works)
python app/orchestrate.py
```

#### Specific Zones

```bash
# Run only the landing zone
python app/cli.py run --stages temporal_landing,persistent_landing

# Run only the formatted zone
python app/cli.py run --stages formatted_documents,formatted_images

# Run only the trusted zone
python app/cli.py run --stages trusted_images,trusted_documents

# Run only the exploitation zone
python app/cli.py run --stages exploitation_documents,exploitation_images

# Run multimodal tasks (after exploitation zone is populated)
python app/cli.py run --stages task1_retrieval,task2_multimodal_search,task3_rag

# Run from formatted zone onwards (faster for development)
python app/cli.py run --stages formatted_documents,formatted_images,trusted_images,trusted_documents,exploitation_documents,exploitation_images
```

#### Single Stages

```bash
# Run a single stage
python app/cli.py run --stage trusted_images
python app/cli.py run --stage exploitation_documents
python app/cli.py run --stage exploitation_images

# Run multimodal tasks individually
python app/cli.py run --stage task1_retrieval
python app/cli.py run --stage task2_multimodal_search
python app/cli.py run --stage task3_rag  # Requires Ollama server running
```

#### Pipeline Options

```bash
# Dry run mode (no actual processing)
python app/cli.py run --dry-run

# Verbose logging
python app/cli.py run --verbose

# Custom configuration file
python app/cli.py run --config custom_pipeline.yaml
```

### CLI Command Reference

| Command    | Description                               |
| ---------- | ----------------------------------------- |
| `run`      | Execute the pipeline                      |
| `status`   | Show pipeline status and available stages |
| `validate` | Validate configuration and environment    |
| `test`     | Run all tests                             |
| `metrics`  | Show real-time system metrics             |
| `report`   | Display latest pipeline execution report  |
| `init`     | Initialize the pipeline environment       |

### Command Options

-   `--stages`: Comma-separated list of stages to run
-   `--stage`: Single stage to run
-   `--dry-run`: Run in dry-run mode (no actual processing)
-   `--verbose`: Enable verbose logging
-   `--config`: Specify configuration file path

## ⚙️ Configuration

### Pipeline Configuration (pipeline.yaml)

The centralized configuration file contains all non-sensitive settings:

```yaml
# Storage configuration
storage:
    buckets:
        landing_zone: "landing-zone"
        formatted_zone: "formatted-zone"
        trusted_zone: "trusted-zone"
    prefixes:
        temporal_landing: "temporal_landing"
        persistent_landing: "persistent_landing"
        formatted_documents: "formatted/documents"
        formatted_images: "formatted/images"
        trusted_documents: "trusted/documents"
        trusted_images: "trusted/images"

# ChromaDB configuration
chromadb_documents:
    collection_name: "exploitation_documents"
    embedding_model: "Qwen/Qwen3-Embedding-0.6B"
    persist_dir: "app/zones/exploitation_zone/chroma_documents"

chromadb_images:
    collection_name: "exploitation_images"
    embedding_model: "ViT-B-32"
    persist_dir: "app/zones/exploitation_zone/chroma_images"

chromadb_multimodal:
    collection_name: "exploitation_multimodal"
    embedding_model: "ViT-B-32"
    persist_dir: "app/zones/exploitation_zone/chroma_exploitation"

# Pipeline settings
pipeline:
    batch_size: 256
    timeout: 3600
    dry_run: false
    overwrite: true
```

## 🐳 Docker Support

The pipeline works with Docker for easy deployment:

```bash
# Start MinIO and other services
cd docker
docker-compose up -d

# Run the pipeline
python app/cli.py run
```

### Docker Services

-   **MinIO**: S3-compatible object storage
-   **Data persistence**: All data stored in `docker/_minio_data/`

### Docker Setup Issues and Solutions

#### MinIO Initialization Script

The `init-minio.sh` script may need to be made executable and have its line endings fixed:

**Make sure the init script is executable (Use Git Bash on Windows or any Unix shell)**:

```bash
chmod +x init-minio.sh
```

**Windows line-endings gotcha (CRLF/BOM)**:
On Windows, the script may fail due to line endings or BOM. Fix it with:

```bash
# Fix line endings and remove BOM
sed -i 's/\r$//' init-minio.sh
sed -i '1s/^\xEF\xBB\xBF//' init-minio.sh
chmod +x init-minio.sh
```

**After fixing the script, re-run**:

```bash
docker compose down -v
docker compose up -d
```

#### Common Docker Issues

1. **Permission Denied on init-minio.sh**

    - Solution: Use Git Bash or WSL on Windows
    - Run: `chmod +x docker/init-minio.sh`

2. **Script Execution Errors**

    - Solution: Fix line endings with the sed commands above
    - Ensure script has Unix line endings (LF, not CRLF)

3. **MinIO Bucket Creation Fails**

    - Solution: Check MinIO logs: `docker logs <minio-container-id>`
    - Verify script permissions and line endings

4. **Port Conflicts**
    - Solution: Change ports in `docker-compose.yml` if 9000/9001 are in use
    - Update `MINIO_ENDPOINT` in your `.env` file accordingly

## 🧪 Testing

The pipeline includes comprehensive unit and integration tests to ensure reliability and correctness.

### Test Structure

```
app/tests/
├── unit/                    # Unit tests for individual components
│   └── test_utils.py        # Tests for shared utilities
└── integration/             # Integration tests for pipeline components
    ├── test_pipeline.py     # Full pipeline integration tests
    └── test_simple_integration.py  # Simple integration tests
```

### Unit Tests (`app/tests/unit/`)

**Purpose**: Test individual components in isolation

**What they test**:

-   **PipelineConfig**: Configuration loading, defaults, nested access, missing keys
-   **Utility Functions**:
    -   `utc_timestamp()`: UTC timestamp generation
    -   `to_builtin()`: Decimal to built-in type conversion
    -   `sanitize_filename()`: Filename sanitization
    -   `atomic_write_json()`: Atomic JSON file writing
-   **Logger**: Logger creation and method functionality
-   **Validation**: Configuration validation with missing/valid environment variables
-   **Multimodal Tasks**:
    -   `Task1RetrievalProcessor`: Text and image retrieval
    -   `ExploitationMultiModalSearcher` (Task 2): Unified multimodal search
    -   `Task3RAGProcessor`: RAG with LLaVA

**Run unit tests**:

```bash
python -m pytest app/tests/unit/ -v

# Run specific task tests
python -m pytest app/tests/unit/test_task1_retrieval.py -v
python -m pytest app/tests/unit/test_task2_multimodal_search.py -v
python -m pytest app/tests/unit/test_task3_rag.py -v
```

### Integration Tests (`app/tests/integration/`)

**Purpose**: Test that pipeline components work together correctly

#### `test_pipeline.py` - Full Pipeline Integration

**What it tests**:

-   **Processor Initialization**: All pipeline processors can be initialized with test configuration
-   **Configuration Consistency**: Same configuration works across all processors
-   **Pipeline Dependencies**: Each stage can access its required configuration
-   **Data Flow**: Correct data flow between pipeline stages (landing → formatted → trusted → exploitation → multimodal tasks)

**Test Coverage**:

-   ✅ `TemporalLandingProcessor` initialization
-   ✅ `PersistentLandingProcessor` initialization
-   ✅ `FormattedDocumentsProcessor` initialization
-   ✅ `FormattedImagesProcessor` initialization
-   ✅ `TrustedImagesProcessor` initialization
-   ✅ `TrustedDocumentsProcessor` initialization
-   ✅ `ExploitationDocumentsProcessor` initialization
-   ✅ `ExploitationImagesProcessor` initialization
-   ✅ `Task1RetrievalProcessor` initialization
-   ✅ `ExploitationMultiModalSearcher` (Task 2) initialization
-   ✅ `Task3RAGProcessor` initialization
-   ✅ Configuration consistency across all processors
-   ✅ Pipeline stage dependencies
-   ✅ Data flow between stages

#### `test_simple_integration.py` - Simple Integration

**What it tests**:

-   **Configuration Loading**: PipelineConfig can load and access configuration
-   **Default Values**: Sensible defaults are provided
-   **Environment Variables**: Environment variables are loaded correctly
-   **Utility Functions**: Shared utility functions work correctly

**Run integration tests**:

```bash
python -m pytest app/tests/integration/ -v
```

### Running All Tests

```bash
# Using CLI (recommended)
python app/cli.py test

# Using pytest directly
python -m pytest app/tests/ -v

# With coverage report
python -m pytest --cov=app app/tests/ --cov-report=html
```

### Test Configuration

Tests use a separate test configuration that includes:

-   **Mock environment variables**: MinIO, Hugging Face, ChromaDB settings
-   **Test buckets and prefixes**: Isolated test storage configuration
-   **Reduced batch sizes**: Faster test execution
-   **Dry run mode**: No actual data processing
-   **Test-specific ChromaDB settings**: Separate test collections

### Test Environment Setup

Tests automatically:

-   Create temporary configuration files
-   Mock external dependencies (MinIO, Hugging Face, ChromaDB)
-   Set up test environment variables
-   Clean up after test completion

### Continuous Integration

The test suite is designed to work in CI/CD environments:

-   No external dependencies required
-   Fast execution (< 30 seconds)
-   Comprehensive coverage of critical paths
-   Clear failure reporting
-   Environment isolation

## 📊 Monitoring and Reporting

### Real-time Monitoring

```bash
# Show system metrics
python app/cli.py metrics

# Show latest execution report
python app/cli.py report
```

### Performance Metrics

The pipeline automatically collects:

-   Execution time per stage
-   Memory usage
-   Disk usage
-   CPU utilization
-   Error tracking
-   Resource monitoring

### Quality Reports

#### Code Quality Report

```bash
# Generate code quality report using pylint
pylint --output-format=text app/ > quality_report.txt

# Or with HTML output
pylint --output-format=html app/ > quality_report.html
```

#### Pipeline Execution Reports

Reports are automatically generated in JSON format:

-   `pipeline_metrics_YYYYMMDD_HHMMSS.json`
-   Contains detailed execution statistics
-   Performance metrics per stage
-   Error summaries

## 📓 Jupyter Notebooks

The `notebooks/` directory contains exploration notebooks for each zone:

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Notebook Structure

-   `landing_zone/`: Raw data exploration
-   `formatted_zone/`: Data cleaning and joining
-   `trusted_zone/`: Quality control analysis
-   `exploitation_zone/`: Vector embeddings and search
-   `fine_tuning/`: Fine-tuning pipeline (dataset preparation, augmentation, experiments)

## 📝 Logging

The pipeline uses structured logging with multiple levels:

-   **INFO**: General pipeline progress
-   **WARNING**: Non-critical issues
-   **ERROR**: Critical errors that may affect processing
-   **DEBUG**: Detailed debugging information

### Log Configuration

```bash
# Set log level via environment
export PIPELINE_LOG_LEVEL=DEBUG

# Or via CLI
python app/cli.py run --verbose
```

### Log Files

-   Console output with timestamps
-   Structured JSON logs for monitoring
-   Error tracking and reporting
-   Performance metrics logging

## 🔍 Troubleshooting

### Common Issues

1. **MinIO Connection Failed**

    ```bash
    # Check MinIO status
    docker ps | grep minio

    # Test connection
    python -c "from app.utils.shared import S3Client; S3Client(PipelineConfig()).test_connection()"
    ```

2. **Hugging Face Authentication**

    ```bash
    # Verify token
    python -c "import os; print('HF_TOKEN set:', bool(os.getenv('HF_TOKEN')))"
    ```

3. **ChromaDB Issues**

    ```bash
    # Check directory permissions
    ls -la app/zones/exploitation_zone/

    # Test ChromaDB
    python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"
    ```

4. **Memory Issues**

    ```bash
    # Monitor memory usage
    python app/cli.py metrics

    # Reduce batch size in pipeline.yaml
    ```

5. **Import Errors**

    ```bash
    # Check Python path
    python -c "import sys; print('Python path:', sys.path)"

    # Test imports
    python -c "from app.utils.config import PipelineConfig; print('Config OK')"
    ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Set debug level
export PIPELINE_LOG_LEVEL=DEBUG

# Run with verbose output
python app/cli.py run --verbose
```

### Windows-Specific Issues

1. **PowerShell Execution Policy**

    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

2. **Path Issues**

    ```bash
    # Use forward slashes in paths
    python app/cli.py run --config app/pipeline.yaml
    ```

3. **Encoding Issues**
    ```bash
    # Set UTF-8 encoding
    chcp 65001
    ```

## 📈 Performance Considerations

### Batch Processing

-   Documents: 256 records per batch (configurable)
-   Images: Individual processing with quality checks
-   ChromaDB embeddings: Batched for efficiency

### Memory Management

-   Streaming processing for large files
-   Multipart uploads for large outputs
-   Memory monitoring and optimization

### Error Handling

-   Comprehensive error handling at each stage
-   Graceful degradation for optional dependencies
-   Detailed error logging and reporting

## 🔄 Continuous Integration

The pipeline is designed to work in CI/CD environments:

-   Automated testing with pytest
-   Configuration validation
-   Performance monitoring
-   Error reporting
-   Quality reports generation

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs for error details
3. Validate your configuration with `python app/cli.py validate`
4. Check the test suite for examples
5. Generate quality reports for code analysis

## 📊 Metrics and Reporting

The pipeline generates comprehensive reports including:

-   Processing statistics per stage
-   Quality metrics (images processed, documents filtered)
-   Performance data (execution time, memory usage)
-   Error summaries and debugging information
-   Resource usage monitoring
-   Code quality analysis

Reports are saved in JSON format and can be integrated with monitoring systems like Grafana, Prometheus, or custom dashboards.

## 🔧 Development

### Adding New Stages

1. Create processor class in appropriate zone folder
2. Implement `process()` method
3. Add to orchestrator stages list
4. Update configuration as needed
5. Add tests

### Configuration Management

-   All configuration in `app/pipeline.yaml`
-   Environment variables for sensitive data
-   Validation with `python app/cli.py validate`

### Code Quality

-   Follow PEP 8 style guidelines
-   Use type hints
-   Comprehensive error handling
-   Unit and integration tests
-   Regular quality reports with pylint

---

## 📚 Additional Resources

### Fine-Tuning Documentation

For detailed information about the fine-tuning pipeline, see:

-   **Fine-Tuning Directory**: `fine_tuning/` contains all notebooks and configuration files
-   **Experiments**: Located in `fine_tuning/experiments/`
-   **Training Data**: Preparation scripts in `fine_tuning/training_data/`
-   **Augmentation**: Augmentation pipeline in `fine_tuning/augmentation/`

All fine-tuning workflows are executed via Jupyter notebooks. Refer to the main README sections above for execution instructions.
