"""
Deep Learning Models Module
============================
Implements LSTM-based deep learning models for trajectory classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


class LSTMTrajectoryModel(nn.Module):
    """LSTM model for ICU trajectory classification."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, output_size: int = 1):
        """
        Initialize LSTM model.
        
        Parameters:
            input_size: Number of features (vital signs)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Output dimension (1 for binary classification)
        """
        super(LSTMTrajectoryModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor (batch_size, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected
        fc_out = self.fc1(last_hidden)
        fc_out = self.relu(fc_out)
        fc_out = self.dropout(fc_out)
        output = self.fc2(fc_out)
        
        return output


class LSTMTrainer:
    """Train and evaluate LSTM models."""
    
    def __init__(self, model: LSTMTrajectoryModel, device: torch.device = None,
                 learning_rate: float = 0.001, random_seed: int = 42,
                 pos_weight: float = None):
        """
        Initialize trainer.

        Parameters:
            model: LSTM model instance
            device: torch device (cuda or cpu)
            learning_rate: Learning rate for optimizer
            random_seed: Random seed for reproducibility
            pos_weight: Weight for positive class in BCEWithLogitsLoss.
                        Set to neg_count/pos_count to handle class imbalance.
                        Without this, LSTM learns to predict all-negative on
                        heavily imbalanced data (e.g. 2-3% sepsis rate).
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Cap pos_weight at 15. 10 → all-negative, 20 → all-positive on this data.
        MAX_POS_WEIGHT = 15.0
        if pos_weight is not None:
            pos_weight_capped = min(pos_weight, MAX_POS_WEIGHT)
            pw = torch.tensor([pos_weight_capped], dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"  pos_weight={pos_weight_capped:.1f} (capped from {pos_weight:.1f})")
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                    weight_decay=1e-4)
        
        self.random_seed = random_seed
        self.train_losses = []
        self.val_losses = []
        
        print(f"\n" + "=" * 60)
        print("DEEP LEARNING - LSTM MODEL")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def prepare_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create torch DataLoaders for training and validation.
        
        Parameters:
            X_train: Training sequences (batch, seq_len, features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Keep tensors on CPU — batches are moved to device in the training loop.
        # Loading everything to GPU upfront causes OOM on large datasets.
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor   = torch.FloatTensor(X_val)
        y_val_tensor   = torch.FloatTensor(y_val).reshape(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

        # Class imbalance handled by pos_weight in BCEWithLogitsLoss; use shuffle only.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
        
        print(f"\nDataLoaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Parameters:
            train_loader: Training DataLoader
            
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model on validation set.
        
        Parameters:
            val_loader: Validation DataLoader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32, patience: int = 15) -> Dict:
        """
        Full training loop with early stopping.
        
        Parameters:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Max number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Dictionary with training info
        """
        train_loader, val_loader = self.prepare_dataloaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        print(f"\nTraining LSTM for {epochs} epochs...")
        print("-" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_model_state)
                break
        
        print(f"? Training complete")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_trained': epoch + 1
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Parameters:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_proba = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = torch.FloatTensor(X_test[i:i+batch_size]).to(self.device)
                logits = self.model(batch)
                all_proba.append(torch.sigmoid(logits).cpu().numpy().flatten())
        y_pred_proba = np.concatenate(all_proba)

        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve': roc_curve(y_test, y_pred_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"\nLSTM Test Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save(self.model.state_dict(), filepath)
        print(f"? Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"? Model loaded from {filepath}")


class GRUTrajectoryModel(nn.Module):
    """GRU model for ICU trajectory classification."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, output_size: int = 1):
        """
        Initialize GRU model.
        
        Parameters:
            input_size: Number of features (vital signs)
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
            output_size: Output dimension (1 for binary classification)
        """
        super(GRUTrajectoryModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor (batch_size, 1)
        """
        # GRU forward
        gru_out, h_n = self.gru(x)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]
        
        # Fully connected
        fc_out = self.fc1(last_hidden)
        fc_out = self.relu(fc_out)
        fc_out = self.dropout(fc_out)
        output = self.fc2(fc_out)
        
        return output


class GRUTrainer:
    """Train and evaluate GRU models."""
    
    def __init__(self, model: GRUTrajectoryModel, device: torch.device = None,
                 learning_rate: float = 0.001, random_seed: int = 42,
                 pos_weight: float = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        MAX_POS_WEIGHT = 15.0
        if pos_weight is not None:
            pos_weight_capped = min(pos_weight, MAX_POS_WEIGHT)
            pw = torch.tensor([pos_weight_capped], dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"  pos_weight={pos_weight_capped:.1f} (capped from {pos_weight:.1f})")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.random_seed = random_seed
        self.train_losses = []
        self.val_losses = []
        
        print(f"\n" + "=" * 60)
        print("DEEP LEARNING - GRU MODEL")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def prepare_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create torch DataLoaders for training and validation.
        
        Parameters:
            X_train: Training sequences (batch, seq_len, features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Keep tensors on CPU — batches are moved to device in the training loop.
        # Loading everything to GPU upfront causes OOM on large datasets.
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor   = torch.FloatTensor(X_val)
        y_val_tensor   = torch.FloatTensor(y_val).reshape(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

        # Class imbalance handled by pos_weight in BCEWithLogitsLoss; use shuffle only.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
        
        print(f"\nDataLoaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Parameters:
            train_loader: Training DataLoader
            
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model on validation set.
        
        Parameters:
            val_loader: Validation DataLoader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32, patience: int = 15) -> Dict:
        """
        Full training loop with early stopping.
        
        Parameters:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Max number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Dictionary with training info
        """
        train_loader, val_loader = self.prepare_dataloaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        print(f"\nTraining GRU for {epochs} epochs...")
        print("-" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_model_state)
                break
        
        print(f"? Training complete")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_trained': epoch + 1
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Parameters:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_proba = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = torch.FloatTensor(X_test[i:i+batch_size]).to(self.device)
                logits = self.model(batch)
                all_proba.append(torch.sigmoid(logits).cpu().numpy().flatten())
        y_pred_proba = np.concatenate(all_proba)

        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve': roc_curve(y_test, y_pred_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"\nGRU Test Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save(self.model.state_dict(), filepath)
        print(f"? Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"? Model loaded from {filepath}")


class BiLSTMTrajectoryModel(nn.Module):
    """Bidirectional LSTM model for ICU trajectory classification."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, output_size: int = 1):
        super(BiLSTMTrajectoryModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        fc_out = self.fc1(last_hidden)
        fc_out = self.relu(fc_out)
        fc_out = self.dropout(fc_out)
        return self.fc2(fc_out)


class BiLSTMTrainer:
    """Train and evaluate BiLSTM models."""
    
    def __init__(self, model: BiLSTMTrajectoryModel, device: torch.device = None,
                 learning_rate: float = 0.001, random_seed: int = 42,
                 pos_weight: float = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        MAX_POS_WEIGHT = 15.0
        if pos_weight is not None:
            pos_weight_capped = min(pos_weight, MAX_POS_WEIGHT)
            pw = torch.tensor([pos_weight_capped], dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"  pos_weight={pos_weight_capped:.1f} (capped from {pos_weight:.1f})")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.random_seed = random_seed
        self.train_losses = []
        self.val_losses = []
        
        print(f"\n" + "=" * 60)
        print("DEEP LEARNING - BILSTM MODEL")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def prepare_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32, patience: int = 15) -> Dict:
        train_loader, val_loader = self.prepare_dataloaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        print(f"\nTraining BiLSTM for {epochs} epochs...")
        print("-" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_model_state)
                break
        
        print(f"? Training complete")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_trained': epoch + 1
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        self.model.eval()
        all_proba = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = torch.FloatTensor(X_test[i:i+batch_size]).to(self.device)
                logits = self.model(batch)
                all_proba.append(torch.sigmoid(logits).cpu().numpy().flatten())
        y_pred_proba = np.concatenate(all_proba)

        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve': roc_curve(y_test, y_pred_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"\nBiLSTM Test Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        return metrics
    
    def save_model(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)
        print(f"? Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"? Model loaded from {filepath}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTrajectoryModel(nn.Module):
    """Transformer encoder for ICU trajectory classification."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.2, output_size: int = 1):
        super(TransformerTrajectoryModel, self).__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        last_hidden = x[:, -1, :]
        fc_out = self.fc1(last_hidden)
        fc_out = self.relu(fc_out)
        fc_out = self.dropout(fc_out)
        return self.fc2(fc_out)


class TransformerTrainer:
    """Train and evaluate Transformer models."""
    
    def __init__(self, model: TransformerTrajectoryModel, device: torch.device = None,
                 learning_rate: float = 0.001, random_seed: int = 42,
                 pos_weight: float = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        MAX_POS_WEIGHT = 15.0
        if pos_weight is not None:
            pos_weight_capped = min(pos_weight, MAX_POS_WEIGHT)
            pw = torch.tensor([pos_weight_capped], dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"  pos_weight={pos_weight_capped:.1f} (capped from {pos_weight:.1f})")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.random_seed = random_seed
        self.train_losses = []
        self.val_losses = []
        
        print(f"\n" + "=" * 60)
        print("DEEP LEARNING - TRANSFORMER MODEL")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def prepare_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32, patience: int = 15) -> Dict:
        train_loader, val_loader = self.prepare_dataloaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        print(f"\nTraining Transformer for {epochs} epochs...")
        print("-" * 60)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = self.model.state_dict()
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_model_state)
                break
        
        print(f"? Training complete")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_trained': epoch + 1
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        self.model.eval()
        all_proba = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = torch.FloatTensor(X_test[i:i+batch_size]).to(self.device)
                logits = self.model(batch)
                all_proba.append(torch.sigmoid(logits).cpu().numpy().flatten())
        y_pred_proba = np.concatenate(all_proba)

        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve': roc_curve(y_test, y_pred_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"\nTransformer Test Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        return metrics
    
    def save_model(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)
        print(f"? Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"? Model loaded from {filepath}")
