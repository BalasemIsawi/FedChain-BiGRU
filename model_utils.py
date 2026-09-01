import os
import random
import hashlib
from datetime import datetime, timezone
import time
import pickle
import gc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import (
    layers, 
    models, 
    callbacks
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, 
    LabelEncoder, 
    MinMaxScaler
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error, 
    mean_absolute_error, 
    silhouette_score
)


#########################################################################
# METHODS
#########################################################################

def prepare_data(csv_path, num_clients=5, iid=True, alpha=0.5, test_size=0.2, SEED=42, verbose=0):
    """
    Prepares TON-IoT dataset and partitions both features (X) and labels (y)
    into parallel client lists for FL.
    """
    dist_type = "IID" if iid else f"Non-IID (Dirichlet alpha={alpha})"
    if verbose !=0 :
        print(f"Loading TON-IoT dataset from: {csv_path}")
        print(f"Partitioning Strategy: {dist_type} across {num_clients} clients.")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # 1. Extract Target Labels
    if 'label' in df.columns:
        y = df['label'].values.ravel().astype(np.int32) # Ensures 1D array shape: (N,)
    else:
        y = None
    
    # 2. Drop high-cardinality metadata & identifiers
    high_cardinality_drops = [
        'src_ip', 'dst_ip', 'src_port', 'dst_port', 
        'http_user_agent', 'http_uri', 'dns_query', 
        'ssl_subject', 'ssl_issuer', 'weird_name', 'weird_addl',
        'label', 'type'
    ]
    X_df = df.drop(columns=[col for col in high_cardinality_drops if col in df.columns], errors='ignore')

    # 3. Clean string booleans & missing values
    pd.set_option('future.no_silent_downcasting', True) # Opt-in to future Pandas behavior
    
    X_df = X_df.replace({'-': np.nan, 'T': 1, 'F': 0, 'true': 1, 'false': 0, True: 1, False: 0})
    X_df = X_df.infer_objects(copy=False) # Explicitly infer updated data types
    X_df = X_df.fillna(0)

    # 4. One-Hot Encode true categoricals
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=float)

    # 5. Drop Zero-Variance columns
    variance = X_df.var(numeric_only=True)
    zero_var_cols = variance[variance == 0].index.tolist()
    if len(zero_var_cols) > 0:
        X_df = X_df.drop(columns=zero_var_cols)

    # 6. Apply Log1p to skewed features
    skewed_cols = X_df.skew(numeric_only=True)
    high_skew = skewed_cols[skewed_cols > 3.0].index.tolist()
    if len(high_skew) > 0:
        X_df[high_skew] = X_df[high_skew].astype(float)
        X_df[high_skew] = np.log1p(np.maximum(0, X_df[high_skew]))

    X = X_df.values.astype(np.float32)

    # 7. Global Train / Test Split
    if y is not None and len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=SEED
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

    if verbose != 0:
        print(f"X_test shape befor scaling {X_test.shape}")
    # 8. Scale Features using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose != 0:
        print(f"X_test shape after scaling {X_test_scaled.shape}")
    # ------------------------------------------------------------------
    # PARALLEL CLIENT PARTITIONING LOGIC (Synchronized X and y)
    # ------------------------------------------------------------------
    X_clients, y_clients = [], []

    if iid:
        # Uniform IID Split along rows
        X_clients = np.array_split(X_train_scaled, num_clients, axis=0)
        y_clients = np.array_split(y_train, num_clients, axis=0) if y_train is not None else [None]*num_clients
    else:
        # Dirichlet Non-IID Partitioning
        N = len(X_train_scaled)
        unique_classes = np.unique(y_train)
        
        """while min_size < 100:
            client_indices = [[] for _ in range(num_clients)]
            for k in unique_classes:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, client_indices)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                client_indices_split = np.split(idx_k, proportions)
                for client_id, idxs in enumerate(client_indices_split):
                    client_indices[client_id].extend(idxs)
            
            min_size = min([len(idxs) for idxs in client_indices])"""
        # Standard, robust Dirichlet Non-IID Partitioning
        min_size = 0
        min_require_size = 10  # Minimum total samples per client to avoid empty splits

        while min_size < min_require_size:
            client_indices = [[] for _ in range(num_clients)]
            for k in unique_classes:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                
                # Sample Dirichlet proportions
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                # Balance splitting points based on proportions
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                client_indices_split = np.split(idx_k, proportions)
                for client_id, idxs in enumerate(client_indices_split):
                    client_indices[client_id].extend(idxs)
                    
            min_size = min([len(idxs) for idxs in client_indices])

        # Slice BOTH X and y with the exact same idxs array
        for client_id in range(num_clients):
            np.random.shuffle(client_indices[client_id])
            idxs = np.array(client_indices[client_id], dtype=int)
            
            X_clients.append(X_train_scaled[idxs])
            if y_train is not None:
                y_clients.append(y_train[idxs])

    return X_clients, y_clients, np.array(X_test_scaled), np.array(y_test)


def build_implicit_temporal_windows(X_data, y_data, sequence_length=10, step=1):
    """
    Converts 2D compressed latent features into 3D temporal sequence arrays.
    Guarantees length synchronization before slicing.
    """
    # 1. Enforce equal sample lengths to prevent out-of-bounds indexing
    min_len = min(len(X_data), len(y_data))
    X_data = X_data[:min_len]
    y_data = y_data[:min_len]
    
    X_seq, y_seq = [], []
    
    # 2. Safely slide window over synchronized arrays
    for i in range(0, min_len - sequence_length + 1, step):
        X_seq.append(X_data[i : i + sequence_length])
        y_seq.append(y_data[i + sequence_length - 1])
        
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)


def evaluate_phase1_autoencoder(
    global_ae, 
    global_encoder, 
    X_test, 
    y_test=None, 
    feature_names=None, 
    client_losses_round_final=None, 
    history_round_losses=None,
    expected_features = 75
):
    """
    Computes and prints Phase 1 Autoencoder metrics.
    Handles unidentical client splits, lists of feature arrays, and transposed inputs.
    """
    results = {}
    
    # ------------------------------------------------------------------
    # 0. INPUT SANITIZATION AND UNIFICATION (Fixes Unidentical Axis Issues)
    # ------------------------------------------------------------------
    # A. Format X_test into a contiguous 2D Matrix of shape (N_samples, N_features)
    print("=" * 65)
    print("      PHASE 1: FEDERATED AUTOENCODER COMPREHENSIVE REPORT      ")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. RECONSTRUCTION FIDELITY METRICS
    # ------------------------------------------------------------------
    # Unified single-pass prediction
    X_recon = global_ae.predict(X_test, batch_size=256, verbose=0)

    global_mse = mean_squared_error(X_test, X_recon)
    global_mae = mean_absolute_error(X_test, X_recon)
    results['Global_MSE'] = global_mse
    results['Global_MAE'] = global_mae

    # B. Per-Feature MAE
    per_feature_mae = np.mean(np.abs(X_test - X_recon), axis=0)
    results['Per_Feature_MAE'] = per_feature_mae

    # C. Feature Correlation Matrix Distortion
    non_const_mask = np.std(X_test, axis=0) > 1e-6
    X_test_filtered = X_test[:, non_const_mask]
    X_recon_filtered = X_recon[:, non_const_mask]

    C_orig = np.corrcoef(X_test_filtered, rowvar=False)
    C_recon = np.corrcoef(X_recon_filtered, rowvar=False)
    
    C_orig = np.nan_to_num(C_orig)
    C_recon = np.nan_to_num(C_recon)
    
    corr_distortion = np.linalg.norm(C_orig - C_recon, ord='fro')
    results['Correlation_Distortion'] = corr_distortion

    print("\n[CATEGORY 1: RECONSTRUCTION FIDELITY]")
    print(f"  • Global Reconstruction MSE : {global_mse:.6f}")
    print(f"  • Global Reconstruction MAE : {global_mae:.6f}")
    print(f"  • Correlation Distortion    : {corr_distortion:.6f} (Frobenius Norm)")
    
    print("\n  • Top 3 Best Reconstructed Features:")
    best_feats = np.argsort(per_feature_mae)[:3]
    for idx in best_feats:
        fname = feature_names[idx] if feature_names and idx < len(feature_names) else f"Feature {idx}"
        print(f"      - {fname}: MAE = {per_feature_mae[idx]:.6f}")

    print("\n  • Top 3 Worst Reconstructed Features:")
    worst_feats = np.argsort(per_feature_mae)[::-1][:3]
    for idx in worst_feats:
        fname = feature_names[idx] if feature_names and idx < len(feature_names) else f"Feature {idx}"
        print(f"      - {fname}: MAE = {per_feature_mae[idx]:.6f}")

    # ------------------------------------------------------------------
    # 2. LATENT SPACE HEALTH & UTILITY METRICS
    # ------------------------------------------------------------------
    Z_latent = global_encoder.predict(X_test, batch_size=256, verbose=0)

    # A. Dimensional Compression Ratio
    input_dim = X_test.shape[1]
    latent_dim = Z_latent.shape[1]
    compression_ratio = (1.0 - (latent_dim / input_dim)) * 100
    results['Compression_Ratio'] = compression_ratio

    # B. Latent Variance / Active Neuron Ratio
    latent_variances = np.var(Z_latent, axis=0)
    active_neurons = np.sum(latent_variances > 1e-4)
    active_ratio = (active_neurons / latent_dim) * 100
    results['Active_Neuron_Ratio'] = active_ratio

    # C. Unsupervised Class Separability (Silhouette Score)
    sil_score = None
    if y_test is not None:
        try:
            _, y_encoded = np.unique(y_test, return_inverse=True)
            unique_classes = np.unique(y_encoded)
            
            if len(unique_classes) >= 2:
                sample_size = min(2000, len(Z_latent))
                train_size = sample_size / len(Z_latent)
                
                Z_sample, _, y_sample, _ = train_test_split(
                    Z_latent, y_encoded, 
                    train_size=train_size, 
                    stratify=y_encoded, 
                    random_state=42
                )
                
                if len(np.unique(y_sample)) >= 2:
                    sil_score = silhouette_score(Z_sample, y_sample)
                    results['Silhouette_Score'] = sil_score
                else:
                    print("  [Warning] Stratified sample contained only 1 class. Skipping Silhouette score.")
            else:
                print(f"  [Warning] y_test contains only {len(unique_classes)} class. Silhouette score requires at least 2.")
        except Exception as e:
            print(f"  [Warning] Could not compute Silhouette score: {e}")

    print("\n[CATEGORY 2: LATENT SPACE HEALTH]")
    print(f"  • Bottleneck Dimension      : {latent_dim} (Original: {input_dim})")
    print(f"  • Compression Ratio         : {compression_ratio:.2f}% reduction")
    print(f"  • Active Bottleneck Neurons : {active_neurons}/{latent_dim} ({active_ratio:.1f}%)")
    if sil_score is not None:
        print(f"  • Latent Silhouette Score   : {sil_score:+.4f} ([-1 to +1], higher is better)")

    # ------------------------------------------------------------------
    # 3. FEDERATED DYNAMICS & CONVERGENCE
    # ------------------------------------------------------------------
    print("\n[CATEGORY 3: FEDERATED DYNAMICS]")
    if client_losses_round_final is not None:
        client_std = np.std(client_losses_round_final)
        client_mean = np.mean(client_losses_round_final)
        results['Client_Loss_Mean'] = client_mean
        results['Client_Loss_Std'] = client_std
        print(f"  • Final Round Client MSE    : {client_mean:.6f} ± {client_std:.6f}")
        print(f"  • Client Drift Variance     : {client_std**2:.8f}")
    else:
        print("  • Final client loss array not passed. (Skipping drift calculation)")

    # ------------------------------------------------------------------
    # 4. COMPUTATIONAL & COMMUNICATION OVERHEAD
    # ------------------------------------------------------------------
    # A. Compression Throughput Speed
    eval_subset = X_test[:min(10000, len(X_test))]
    start_time = time.time()
    _ = global_encoder.predict(eval_subset, batch_size=256, verbose=0)
    end_time = time.time()
    throughput_ms = ((end_time - start_time) / len(eval_subset)) * 1000
    results['Throughput_ms_per_sample'] = throughput_ms

    # B. Model Parameter Bandwidth per FL Round
    total_ae_params = global_ae.count_params()
    payload_mb_per_client = (total_ae_params * 4) / (1024 * 1024)
    results['Payload_MB_per_Client'] = payload_mb_per_client

    print("\n[CATEGORY 4: COMPUTATION & COMMUNICATION]")
    print(f"  • Encoder Throughput       : {throughput_ms:.4f} ms/sample")
    print(f"  • Total AE Parameters      : {total_ae_params:,}")
    print(f"  • FL Payload Size / Client : {payload_mb_per_client:.3f} MB per round")
    print("=" * 65 + "\n")

    return results


def plot_federated_convergence(round_losses):
    """Plots Phase 1 FL Convergence Curve"""
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(round_losses) + 1), round_losses, marker='o', linewidth=2, color='#1f77b4')
    plt.title("Phase 1: Federated Autoencoder Reconstruction Loss Convergence", fontsize=12)
    plt.xlabel("FL Communication Round", fontsize=10)
    plt.ylabel("Average Local Reconstruction MSE", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def compute_binary_metrics(y_true, y_pred_probs, threshold=0.5):
    """Calculates comprehensive classification metrics for binary classification."""
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Calculate scalar metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC requires at least two classes present in y_true
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        roc_auc = 0.5  # Default fallback if a client only has 1 class in validation

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }


def compute_robust_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Equal weight to Normal (0) and Attack (1) detection:
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    
    return balanced_accuracy, f1_macro


def federated_averaging(weights_list, sample_counts):
    """Weighted average of model parameters across clients (FedAvg)"""
    total_samples = sum(sample_counts)
    aggregated_weights = []

    # Iterate through layer weights
    for layer_idx in range(len(weights_list[0])):
        weighted_layer = np.zeros_like(weights_list[0][layer_idx], dtype=np.float32)
        for client_idx in range(len(weights_list)):
            weight = sample_counts[client_idx] / total_samples
            weighted_layer += weights_list[client_idx][layer_idx] * weight
        aggregated_weights.append(weighted_layer)

    return aggregated_weights


def serialize_weights(weights):
    """
    Includes tensor shapes and bytes so different tensor layouts
    cannot accidentally produce the same byte sequence.
    """
    chunks = []

    for w in weights:
        shape_bytes = str(w.shape).encode("utf-8")
        chunks.append(shape_bytes)
        chunks.append(w.astype(np.float32).tobytes())

    return b"".join(chunks)


def hash_model_weights(weights):
    return hashlib.sha256(serialize_weights(weights)).hexdigest()


def evaluate_binary_model(model, X_test_seq, y_test_seq, threshold=0.5):
    probabilities = model.predict(X_test_seq, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(np.int32)

    # ROC-AUC requires at least two classes present in y_test_seq
    try:
        roc_auc = roc_auc_score(y_test_seq, predictions)
    except ValueError:
        roc_auc = 0.5  # Default fallback if a client only has 1 class in validation

    results = {
        "accuracy": accuracy_score(y_test_seq, predictions),
        "precision": precision_score(y_test_seq, predictions, zero_division=0),
        "recall": recall_score(y_test_seq, predictions, zero_division=0),
        "f1_score": f1_score(y_test_seq, predictions, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test_seq, predictions),
        "classification_report": classification_report(
            y_test_seq,
            predictions,
            target_names=["Normal", "Attack"],
            zero_division=0
        )
    }

    return results


def compute_binary_metrics(y_true, y_pred_probs, threshold=0.5):
    """Calculates comprehensive classification metrics for binary classification."""
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Calculate scalar metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC requires at least two classes present in y_true
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        roc_auc = 0.5  # Default fallback if a client only has 1 class in validation

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }


########################################################################
# CLASSES
#########################################################################

class ClientAE:
    def __init__(self, client_id, X_local, input_dim, bottleneck_dim, ae):
        self.client_id = client_id
        self.X_local = X_local # Local 2D tabular features (unlabeled)
        self.autoencoder, _ = ae(input_dim, bottleneck_dim)

    def set_weights(self, global_weights):
        """Update local AE model with weights from Central Server"""
        self.autoencoder.set_weights(global_weights)

    def train_local(self, local_epochs=5, batch_size=128):
        """Train Autoencoder on local dataset"""
        history = self.autoencoder.fit(
            self.X_local, self.X_local,
            epochs=local_epochs,
            batch_size=batch_size,
            verbose=0
        )
        local_loss = history.history['loss'][-1]
        # Return updated weights and number of samples
        return self.autoencoder.get_weights(), len(self.X_local), local_loss


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class TemporalAttention(layers.Layer):
    """Computes a context vector as a weighted sum of BiGRU sequence outputs."""
    def __init__(self, **kwargs):
        # Direct class initialization prevents super() module-reload type mismatches
        layers.Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", 
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform", 
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias", 
            shape=(input_shape[1], 1),
            initializer="zeros", 
            trainable=True
        )
        layers.Layer.build(self, input_shape)

    def call(self, inputs):
        # Score calculation: e = tanh(X * W + b)
        e = tf.math.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # Attention weights: alpha = softmax(e)
        alpha = tf.nn.softmax(e, axis=1)
        # Context vector: sum(inputs * alpha)
        context = tf.reduce_sum(inputs * alpha, axis=1)
        return context

    def get_config(self):
        config = super().get_config() if hasattr(super(), "get_config") else layers.Layer.get_config(self)
        return config

class ModelUpdateBlockchain:
    """
    Research prototype:
    - stores only hashes and metadata
    - does not store raw data or full model weights on-chain
    """
    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = {
            "index": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": 0,
            "model_hash": "GENESIS",
            "previous_hash": "0"
        }
        genesis["block_hash"] = self._calculate_block_hash(genesis)
        self.chain.append(genesis)

    @staticmethod
    def _calculate_block_hash(block):
        payload = (
            f"{block['index']}|{block['timestamp']}|{block['round']}|"
            f"{block['model_hash']}|{block['previous_hash']}"
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()

    def add_model_update(self, round_number, aggregated_weights):
        model_hash = hash_model_weights(aggregated_weights)
        previous_hash = self.chain[-1]["block_hash"]

        block = {
            "index": len(self.chain),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": int(round_number),
            "model_hash": model_hash,
            "previous_hash": previous_hash
        }

        block["block_hash"] = self._calculate_block_hash(block)
        self.chain.append(block)

        return block

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block["previous_hash"] != previous_block["block_hash"]:
                return False

            recalculated_hash = self._calculate_block_hash(current_block)
            if recalculated_hash != current_block["block_hash"]:
                return False

        return True


