# FedChain-BiGRU: Blockchain-Based Federated Learning for IoT Intrusion Detection

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: Research](https://img.shields.io/badge/License-Research-blue.svg)](#license)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-red.svg)](#project-status)

A research project implementing federated learning with blockchain integration for distributed intrusion detection on the TON-IoT dataset. This implementation combines **Blockchain Technology** with **Federated Learning (FL)** to create a secure, decentralized framework for training machine learning models on IoT intrusion detection. The approach enables multiple clients (IoT devices/organizations) to collaboratively train models without centralizing sensitive data.

## 🎯 Project Overview

This project bridges two critical technologies in distributed machine learning:
- **Federated Learning**: Enables collaborative model training across multiple clients while preserving data privacy
- **Blockchain Integration**: Provides immutable record-keeping, verification, and audit trails for model aggregations

### ✨ Key Features

- 🔐 **Federated Learning**: Distributed training across multiple clients with data privacy preservation
- ⛓️ **Blockchain Integration**: Secure model aggregation and verification using blockchain
- 📊 **TON-IoT Dataset**: Real-world Internet of Things security dataset for intrusion detection
- 🧠 **Deep Learning Models**: BiGRU (Bidirectional Gated Recurrent Unit) neural networks with baseline comparisons
- 🔀 **Flexible Data Partitioning**: Support for both IID (Independent & Identically Distributed) and Non-IID data distributions
- 📈 **Comprehensive Metrics**: Accuracy, F1-score, ROC-AUC, precision, recall, and confusion matrices
- 📝 **Reproducibility**: Full experiment logging, metrics tracking, and configurable hyperparameters

## 📁 Project Structure

```
FedChain-BiGRU/
├── README.md                              # This file - Project documentation
├── __init__.py                            # Python package initialization
├── model_utils.py                         # Core utility functions for data preparation and model training
│
├── 📓 Jupyter Notebooks/
│   ├── fed_chain_bigru.ipynb             # Main federated learning experiments with BiGRU model
│   ├── blockchain_tests.ipynb            # Blockchain integration and verification tests
│   ├── baseline_models_comparison.ipynb  # Comparison with centralized baseline models
│   ├── experiments_metrics.ipynb          # Metrics collection and analysis
│   ├── converg_results.ipynb             # Convergence analysis and visualization
│   ├── extract_results.ipynb             # Results extraction and post-processing
│   └── evaluate_data_splits.ipynb        # Data partitioning validation and analysis
│
├── ton-iot-train-test-network/           # Neural network testing environment
│   └── download.md                       # Dataset download instructions
│
├── experiments/                          # Experiment configurations and results
├── logs/                                 # Training logs and debug information
└── __pycache__/                          # Python cache files
```

## 📋 Requirements & Dependencies

### Core Libraries
- **TensorFlow/Keras** (2.x+): Deep learning framework for neural networks
- **Scikit-learn** (1.x+): Machine learning utilities and evaluation metrics
- **NumPy & Pandas**: Data manipulation and numerical computing
- **Matplotlib**: Data visualization and result plotting
- **Jupyter Notebook**: Interactive data analysis and experimentation

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large-scale experiments)
- **Storage**: At least 10GB for dataset and results

## 🚀 Installation & Setup

### Prerequisites
Before getting started, ensure you have:
- Python 3.8+ installed
- pip or conda package manager
- Git for version control

### Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/BalasemIsawi/FedChain-BiGRU.git
cd FedChain-BiGRU
```

**2. Create a virtual environment (recommended):**
```bash
# Using conda
conda create -n fedchain-fl python=3.9
conda activate fedchain-fl

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
# Using pip
pip install tensorflow scikit-learn numpy pandas matplotlib jupyter

# Or using conda
conda install -c conda-forge tensorflow scikit-learn numpy pandas matplotlib jupyter
```

**4. Verify installation:**
```python
python -c "import tensorflow; import sklearn; print('✓ Setup complete!')"
```

## 💻 Usage & Running Experiments

### Starting Jupyter Lab/Notebook

```bash
jupyter notebook
# or
jupyter lab
```

Then open any `.ipynb` file to begin experimenting.

### Data Preparation

The `model_utils.py` module provides comprehensive data preparation utilities. Here's how to use it:

```python
from model_utils import prepare_data

# Load and prepare TON-IoT dataset with client partitioning
X_train_clients, X_test, y_train_clients, y_test = prepare_data(
    csv_path='path/to/ton-iot.csv',
    num_clients=5,              # Number of federated clients
    iid=True,                   # IID data distribution (True) or Non-IID (False)
    alpha=0.5,                  # Dirichlet concentration parameter (for non-IID)
    test_size=0.2,              # Proportion of test data
    SEED=42,                    # Random seed for reproducibility
    verbose=1                   # Print progress information
)
```

### Running Experiments

#### 1. Federated Learning with BiGRU
Open and run `fed_chain_bigru.ipynb` to execute federated learning training:
- Implements BiGRU model architecture
- Trains models across multiple simulated clients
- Aggregates model updates
- Tracks convergence and metrics

#### 2. Baseline Comparison
Run `baseline_models_comparison.ipynb` to compare:
- Federated learning approach vs. centralized models
- Different model architectures (LSTM, GRU, Dense networks)
- Performance trade-offs

#### 3. Blockchain Verification
Execute `blockchain_tests.ipynb` to validate:
- Blockchain-based model aggregation
- Verification of federated updates
- Tamper-proof model history

#### 4. Results Analysis
- **experiments_metrics.ipynb**: Analyze training metrics and performance
- **converg_results.ipynb**: Study model convergence behavior
- **evaluate_data_splits.ipynb**: Validate data partitioning strategies
- **extract_results.ipynb**: Post-process and extract results

### Advanced Configuration

#### Data Distribution Strategies

**IID Distribution** (uniform distribution):
```python
prepare_data(..., iid=True)
```

**Non-IID Distribution** (heterogeneous data):
```python
prepare_data(..., iid=False, alpha=0.5)
```
- Lower alpha values create more heterogeneous (non-IID) distributions
- Higher alpha values create more homogeneous (IID) distributions

## 📊 Dataset: TON-IoT Network

### Overview
- **Name**: TON-IoT (Internet of Things) Network Dataset
- **Source**: Real-world IoT network traffic from multiple organizations
- **Purpose**: Intrusion detection and anomaly classification
- **Classes**: Binary (Normal vs. Attack) or Multi-class (specific attack types)
- **Scale**: Hundreds of thousands to millions of flow records

### Dataset Characteristics
- **Network Flows**: Captures TCP/UDP session information
- **Protocol Information**: Application layer protocols (HTTP, HTTPS, DNS, etc.)
- **Packet Statistics**: Flow duration, packet counts, byte counts
- **Bidirectional Features**: Forward and backward direction analysis

### Data Preprocessing Pipeline

The project applies comprehensive preprocessing:
1. **Feature Cleaning**:
   - Remove high-cardinality features (IP addresses, ports, URIs)
   - Handle missing values
   - Remove duplicate rows

2. **Encoding**:
   - Boolean encoding (T/F, true/false → 0/1)
   - One-hot encoding for categorical variables
   - Label encoding for target classes

3. **Handling Imbalance**:
   - Log transformation for skewed distributions
   - Class weighting in model training

4. **Normalization**:
   - StandardScaler for zero-mean unit-variance
   - MinMaxScaler for [0,1] scaling

### Downloading the Dataset

#### Option 1: Kaggle
```python
import pandas as pd
file_path = '/kaggle/input/datasets/arnobbhowmik/ton-iot-network-dataset/train_test_network.csv'
df = pd.read_csv(file_path, nrows=500000)
print(f"Data Shape: {df.shape}")
```

#### Option 2: UNSW Canberra
Visit: [UNSW TON-IoT Datasets](https://research.unsw.edu.au/projects/toniot-datasets)

For detailed download instructions, see [ton-iot-train-test-network/download.md](./ton-iot-train-test-network/download.md)

## 🔧 Key Components

### model_utils.py
Core module containing essential utility functions:

#### Data Preparation (`prepare_data`)
- CSV loading and preprocessing
- Client data partitioning for federated learning
- IID and Non-IID distribution support
- Feature engineering and normalization
- Train-test splitting with stratification

#### Model Architecture
- Sequential model building utilities
- BiGRU (Bidirectional GRU) implementation
- LSTM and Dense layer variants
- Regularization (dropout, L2)

#### Training & Evaluation
- Federated training loops
- Local model updates
- Model aggregation strategies
- Comprehensive metrics computation

### Federated Learning Framework

#### Local Training Phase
- Each client trains on local data independently
- Preserves data privacy (raw data never leaves the client)
- Customizable training epochs and batch sizes

#### Model Aggregation Phase
- Collects local model updates
- Performs secure aggregation
- Blockchain verification of aggregation
- Generates new global model

#### Communication
- Efficient parameter averaging
- Configurable communication rounds
- Optional compression of model updates

### Blockchain Integration

Provides:
- ⛓️ **Immutable Records**: All model aggregations recorded permanently
- ✓ **Verification**: Cryptographic verification of federated updates
- 🛡️ **Tamper Detection**: Detect unauthorized model modifications
- 📋 **Audit Trail**: Complete history for compliance and transparency

See `blockchain_tests.ipynb` for detailed blockchain functionality and testing.

## 📈 Metrics & Evaluation

### Classification Performance Metrics
- **Accuracy**: Overall correctness of predictions
- **Balanced Accuracy**: Average of recall scores for each class
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of prediction types

### Convergence Analysis
- **Model Loss**: Training and validation loss per round
- **Accuracy Trends**: Improvement of accuracy over federated rounds
- **Gradient Statistics**: Movement and magnitude of weight updates
- **Communication Rounds**: Convergence speed and efficiency

### System Performance Metrics
- **Training Time**: Execution time per round and total
- **Communication Overhead**: Data transmitted between clients and server
- **Memory Usage**: Peak and average memory consumption
- **Throughput**: Samples processed per second

## 📊 Results & Outputs

The project generates comprehensive results including:
- **Training Curves**: Loss and accuracy plots over epochs and rounds
- **Convergence Analysis**: Visualization of federated convergence
- **Confusion Matrices**: Attack classification breakdowns
- **Performance Comparison Tables**: Federated vs. centralized approaches
- **Blockchain Verification Logs**: Model aggregation audit trail
- **Experiment Reports**: Detailed metrics and configurations

Results are automatically saved to:
- `experiments/`: Experiment configurations and numerical results
- `logs/`: Training logs and debugging information
- Generated plots and visualizations

## 🔄 Experimental Configurations

### Data Partitioning Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **IID** | Uniform distribution across clients | Baseline comparisons |
| **Non-IID (α=0.5)** | Heterogeneous, realistic distribution | Real-world scenarios |
| **Non-IID (α=0.1)** | Highly heterogeneous data | Robustness testing |

### Model Configurations

| Component | Setting | Description |
|-----------|---------|-------------|
| **Architecture** | BiGRU | Bidirectional Gated Recurrent Unit |
| **Activation** | ReLU, Sigmoid | Non-linear transformations |
| **Optimizer** | Adam | Adaptive learning rate optimization |
| **Loss** | Cross-Entropy | Binary or categorical classification |
| **Regularization** | Dropout, L2 | Overfitting prevention |

### Training Parameters

```python
# Example training configuration
config = {
    'num_clients': 5,              # Number of federated clients
    'num_rounds': 50,              # Federated training rounds
    'epochs_per_client': 3,        # Local training epochs
    'batch_size': 32,              # Batch size for training
    'learning_rate': 0.001,        # Initial learning rate
    'dropout_rate': 0.3,           # Dropout probability
    'test_size': 0.2,              # Test set proportion
    'random_seed': 42              # Reproducibility
}
```

## 🤝 Contributing

This is a research project. Contributions and feedback are welcome!

### To Contribute:
1. Review existing notebooks for context and coding style
2. Check `experiments/` for configuration examples
3. Refer to `model_utils.py` for API and function signatures
4. Create a new branch for your changes
5. Submit a pull request with a clear description

### Research Contributions:
- New model architectures
- Alternative aggregation strategies
- Blockchain optimizations
- Privacy-preserving techniques
- Performance improvements

## 📚 References & Related Work

### Key Publications
- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" ([arXiv:1602.05629](https://arxiv.org/abs/1602.05629))
- **Blockchain for ML**: Integration of distributed ledger technology with machine learning models
- **IoT Security**: TON-IoT intrusion detection challenges and methodologies

### Datasets
- **TON-IoT**: [UNSW Canberra Research](https://research.unsw.edu.au/projects/toniot-datasets)
- **Kaggle**: [TON-IoT on Kaggle](https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset)

### Related Technologies
- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
- Federated Learning: [TensorFlow Federated](https://www.tensorflow.org/federated)

## 📝 Citation

If you use this project in your research or work, please cite it as:

```bibtex
@misc{fedchain-bigru,
  title={FedChain-BiGRU: Blockchain-Based Federated Learning for IoT Intrusion Detection},
  author={Isawi, Balasem},
  year={2026},
  publisher={GitHub},
  howpublished{\url{https://github.com/BalasemIsawi/FedChain-BiGRU}}
}
```

## 📄 License

This project is provided for research and educational purposes. 

**Please check with the repository owner regarding specific licensing details.**

For the TON-IoT dataset, please refer to the original dataset's license terms.

## ⚠️ Disclaimer

This is an experimental research implementation. While we strive for correctness:
- Results may vary based on environment and configuration
- No warranty for production use without thorough testing
- Always validate results independently
- Follow responsible AI practices when deploying

## 📧 Support & Acknowledgments

### Acknowledgments
- TON-IoT Dataset contributors and UNSW Canberra team
- Federated Learning and Blockchain research communities
- TensorFlow and Scikit-learn open-source communities
- Research partners and institutions

### Contact
For questions, issues, or suggestions:
- Open an issue on [GitHub Issues](https://github.com/BalasemIsawi/FedChain-BiGRU/issues)
- Contact the repository owner

### Getting Help
- Review Jupyter notebooks for usage examples
- Check `model_utils.py` for function documentation
- Examine `experiments/` for configuration examples
- Refer to inline comments in code files

## 📌 Project Status & Timeline

| Milestone | Status | Date |
|-----------|--------|------|
| Initial Implementation | ✅ Complete | Sept 2026 |
| Blockchain Integration | ✅ Complete | Sept 2026 |
| Performance Evaluation | ✅ Complete | Sept 2026 |
| Documentation | ✅ Complete | Sept 2026 |
| Future Enhancements | 🔄 Planned | TBD |

---

**Last Updated**: September 2026  
**Project Status**: Research/Experimental  
**Python Version**: 3.8+  
**Framework**: TensorFlow 2.x, Scikit-learn 1.x  
**Maintained by**: [@BalasemIsawi](https://github.com/BalasemIsawi)

**⭐ If you find this project useful, please consider starring it on GitHub!**
