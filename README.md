# Blockchain-Based Federated Learning for IoT Intrusion Detection

A research project implementing federated learning with blockchain integration for distributed intrusion detection on the TON-IoT dataset.

## Project Overview

This project combines **Blockchain Technology** with **Federated Learning (FL)** to create a secure, decentralized framework for training machine learning models on IoT intrusion detection. The approach enables multiple clients (IoT devices/organizations) to collaboratively train models without centralizing sensitive data.

### Key Features

- **Federated Learning**: Distributed training across multiple clients with data privacy preservation
- **Blockchain Integration**: Secure model aggregation and verification using blockchain
- **TON-IoT Dataset**: Real-world Internet of Things security dataset for intrusion detection
- **Deep Learning Models**: BiGRU-based neural networks and baseline comparisons
- **Flexible Data Partitioning**: Support for both IID (Independent & Identically Distributed) and Non-IID data distributions
- **Comprehensive Metrics**: Accuracy, F1-score, ROC-AUC, precision, recall, and confusion matrices

## Project Structure

```
block_chain_federated_3/
├── README.md                           # This file
├── model_utils.py                      # Core utility functions for data preparation and model training
├── __init__.py                         # Python package initialization
│
├── Jupyter Notebooks/
│   ├── fed_chain_bigru.ipynb          # Main federated learning experiments with BiGRU model
│   ├── blockchain_tests.ipynb         # Blockchain integration and verification tests
│   ├── baseline_models_comparison.ipynb # Comparison with centralized baseline models
│   ├── experiments_metrics.ipynb       # Metrics collection and analysis
│   ├── converg_results.ipynb          # Convergence analysis and visualization
│   ├── extract_results.ipynb          # Results extraction and post-processing
│   └── evaluate_data_splits.ipynb     # Data partitioning validation and analysis
│
├── ton-iot-train-test-network/        # Neural network testing environment
├── experiments/                        # Experiment configurations and results
├── logs/                              # Training logs and debug information
└── __pycache__/                       # Python cache files
```

## Dependencies

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework for neural networks
- **Scikit-learn**: Machine learning utilities and metrics
- **NumPy & Pandas**: Data manipulation and numerical computing
- **Matplotlib**: Data visualization

### Development
- **Jupyter Notebook**: Interactive data analysis and experimentation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone or download the project:
```bash
cd block_chain_federated_3
```

2. Install required dependencies:
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib jupyter
```

Or using conda:
```bash
conda create -n blockchain-fl python=3.9
conda activate blockchain-fl
conda install -c conda-forge tensorflow scikit-learn numpy pandas matplotlib jupyter
```

## Usage

### Data Preparation

The `model_utils.py` module provides data preparation utilities:

```python
from model_utils import prepare_data

# Load and prepare TON-IoT dataset
X_train_clients, X_test, y_train_clients, y_test = prepare_data(
    csv_path='path/to/ton-iot.csv',
    num_clients=5,           # Number of federated clients
    iid=True,               # IID data distribution
    alpha=0.5,              # Dirichlet concentration (for non-IID)
    test_size=0.2,          # Test set proportion
    SEED=42,                # Random seed for reproducibility
    verbose=1               # Print progress
)
```

### Running Experiments

1. **Federated Learning with BiGRU**:
   Open and run `fed_chain_bigru.ipynb` to execute federated learning training with BiGRU model architecture.

2. **Baseline Comparison**:
   Run `baseline_models_comparison.ipynb` to compare federated approach against centralized models.

3. **Blockchain Verification**:
   Execute `blockchain_tests.ipynb` to validate blockchain-based model aggregation.

4. **Results Analysis**:
   - `experiments_metrics.ipynb`: Analyze training metrics
   - `converg_results.ipynb`: Study convergence behavior
   - `evaluate_data_splits.ipynb`: Validate data partitioning

## Dataset

### TON-IoT Dataset
- **Source**: Real-world Internet of Things network traffic data
- **Purpose**: Intrusion detection classification
- **Classes**: Normal traffic vs. various attack types
- **Features**: Network flow features, protocol information, packet statistics
- **Preprocessing**: 
  - High-cardinality features removal (IPs, ports, URIs)
  - Boolean encoding (T/F, true/false)
  - One-hot encoding for categorical variables
  - Handling skewed distributions with log transformation
  - Feature normalization and scaling

## Key Components

### `model_utils.py`

Contains essential functions for:

- **Data Preparation** (`prepare_data`):
  - CSV loading and preprocessing
  - Client data partitioning (IID and Non-IID)
  - Feature engineering and cleaning
  - Train-test splitting

- **Model Architecture**: Building blocks for neural networks

- **Training & Evaluation**: Federated training and metrics computation

### Federated Learning Framework

- **Local Training**: Each client trains on local data
- **Model Aggregation**: Blockchain-verified aggregation of local models
- **Privacy Preservation**: Data never leaves client devices
- **Communication Efficiency**: Selective parameter averaging

## Metrics & Evaluation

The project tracks multiple performance metrics:

- **Classification Metrics**:
  - Accuracy
  - Balanced Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC Score
  - Confusion Matrix

- **Convergence Metrics**:
  - Model loss over rounds
  - Accuracy improvement trends
  - Gradient statistics

- **System Metrics**:
  - Training time
  - Communication overhead
  - Memory usage

## Results

The project generates:
- Training metrics and loss curves
- Convergence analysis plots
- Confusion matrices for attack classification
- Performance comparison tables
- Blockchain verification logs

Results are saved in the `experiments/` and `logs/` directories.

## Blockchain Integration

The blockchain component provides:
- Immutable record of model aggregations
- Verification of federated updates
- Tamper-proof model history
- Audit trail for compliance

See `blockchain_tests.ipynb` for detailed blockchain functionality.

## Experimental Setup

### Data Partitioning Strategies

1. **IID Distribution**: Data uniformly distributed across clients
   ```python
   prepare_data(..., iid=True)
   ```

2. **Non-IID Distribution**: Data distributed using Dirichlet process
   ```python
   prepare_data(..., iid=False, alpha=0.5)
   ```
   - Lower alpha values create more heterogeneous distributions

### Model Configuration

- **Architecture**: BiGRU (Bidirectional Gated Recurrent Unit)
- **Activation Functions**: ReLU, Sigmoid
- **Optimizer**: Adam
- **Loss Function**: Binary/Categorical Cross-Entropy
- **Regularization**: Dropout, L2 regularization

## Contributing

This is a research project. For questions or contributions:
1. Review existing notebooks for context
2. Check `experiments/` for configuration examples
3. Refer to `model_utils.py` for utility function signatures

## Citation

If you use this project in your research, please cite appropriately.

## License

This project is provided for research purposes. Please check with the repository owner for licensing details.

## Acknowledgments

- TON-IoT Dataset contributors
- Federated Learning and Blockchain research communities
- TensorFlow and scikit-learn communities

## Contact & Support

For questions about this implementation, refer to:
- Jupyter notebooks for usage examples
- `model_utils.py` for API documentation
- `experiments/` directory for configuration examples

## References

### Key Papers & Concepts

- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **Blockchain for ML**: Integration of distributed ledger technology with machine learning
- **IoT Security**: TON-IoT intrusion detection challenges and methodologies

---

**Last Updated**: September 2026
**Project Status**: Research/Experimental
**Python Version**: 3.8+
**Framework**: TensorFlow 2.x, Scikit-learn 1.x
