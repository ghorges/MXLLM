"""
Deep Neural Network (DNN) Algorithm
TensorFlow/Keras implementation for deep learning
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
import time
from .base_algorithm import BaseAlgorithm

try:
    import tensorflow as tf
    from tensorflow import keras
    DNN_AVAILABLE = True
except ImportError:
    print("Warning: tensorflow not installed, DNN algorithm unavailable")
    DNN_AVAILABLE = False


class DNNAlgorithm(BaseAlgorithm):
    def __init__(self, epochs=500, batch_size=8, learning_rate=0.005, 
                 dropout_rate=0.1, use_grid_search=False):
        """Initialize DNN algorithm"""
        super().__init__("DNN")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.use_grid_search = use_grid_search
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'DNNAlgorithm':
        """Train DNN model"""
        if not DNN_AVAILABLE:
            raise ImportError("tensorflow not installed, cannot use DNN algorithm")
        
        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train)
        X_train_scaled, _ = self._ensure_standardized(X_train_processed)
        
        input_dim = X_train_scaled.shape[1]
        n_classes = len(np.unique(y_train_processed))
        
        self.model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_dim,),
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(self.dropout_rate * 0.5),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        loss = 'binary_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=50,
                restore_best_weights=True,
                monitor='val_accuracy',
                mode='max',
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.3,
                patience=20,
                monitor='val_accuracy',
                mode='max',
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        self.model.fit(
            X_train_scaled, y_train_processed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X_test_processed, _ = self.prepare_data(X_test)
        X_test_scaled, _ = self._ensure_standardized(X_test_processed)
        
        predictions = self.model.predict(X_test_scaled, verbose=0)
        
        # For binary classification with sigmoid output
        return (predictions.flatten() > 0.5).astype(int) 