"""
Deep Learning Model for Exoplanet Atmospheric Composition Prediction
Uses Transformer-inspired architecture with attention mechanisms
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Reshape for attention: (batch, 1, features)
        inputs_expanded = tf.expand_dims(inputs, 1)
        attn_output = self.att(inputs_expanded, inputs_expanded)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs_expanded + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        # Squeeze back to original shape
        return tf.squeeze(out2, 1)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ExoplanetAtmosphereModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_molecules = ['H2', 'He', 'H2O', 'CO', 'CH4', 'CO2', 'NH3', 'N2']
        
    def build_model(self, input_dim, output_dim):
        """Build transformer-based neural network"""
        
        inputs = layers.Input(shape=(input_dim,))
        
        # Initial embedding
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.2)(x)
        x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.2)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer with softmax to ensure percentages sum to 100
        outputs = layers.Dense(output_dim, activation='softmax')(x)
        
        # Scale outputs to percentage (multiply by 100)
        outputs = layers.Lambda(lambda x: x * 100)(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Custom loss: MAE + constraint that sum should be ~100
        def custom_loss(y_true, y_pred):
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            sum_constraint = tf.reduce_mean(tf.square(tf.reduce_sum(y_pred, axis=1) - 100))
            return mae + 0.1 * sum_constraint
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_data(self, dataset_path='exoplanet_dataset.csv'):
        """Load and prepare data for training"""
        
        df = pd.read_csv(dataset_path)
        
        # Load feature names
        self.feature_names = joblib.load('feature_names.pkl')
        
        # Separate features and targets
        X = df[self.feature_names].values
        y = df[self.target_molecules].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, dataset_path='exoplanet_dataset.csv', epochs=100):
        """Train the model"""
        
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(dataset_path)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Input features: {X_train.shape[1]}")
        print(f"Output molecules: {y_train.shape[1]}")
        
        # Build model
        print("\nBuilding model...")
        self.model = self.build_model(X_train.shape[1], y_train.shape[1])
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating model...")
        test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test)
        print(f"Test MAE: {test_mae:.2f}%")
        print(f"Test MSE: {test_mse:.2f}")
        
        # Save model and scaler
        self.save_model()
        
        return history
    
    def save_model(self):
        """Save model and preprocessing objects"""
        self.model.save('exoplanet_atmosphere_model.keras')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        joblib.dump(self.target_molecules, 'target_molecules.pkl')
        print("\nModel saved successfully!")
    
    def load_model(self):
        """Load trained model and preprocessing objects"""
        self.model = keras.models.load_model(
            'exoplanet_atmosphere_model.keras',
            custom_objects={'TransformerBlock': TransformerBlock}
        )
        self.scaler = joblib.load('scaler.pkl')
        self.feature_names = joblib.load('feature_names.pkl')
        self.target_molecules = joblib.load('target_molecules.pkl')
        print("Model loaded successfully!")
    
    def predict(self, features_dict):
        """Predict atmospheric composition for a single exoplanet"""
        
        # Extract features in correct order
        features = []
        for name in self.feature_names:
            if name in features_dict:
                features.append(features_dict[name])
            elif name == 'insolation_flux':
                # Calculate derived feature
                st_rad = features_dict.get('st_rad', 1.0)
                pl_orbper = features_dict.get('pl_orbper', 365)
                features.append((st_rad ** 2) / (pl_orbper ** (2/3)))
            elif name == 'surface_gravity':
                pl_bmasse = features_dict.get('pl_bmasse', 1.0)
                pl_rade = features_dict.get('pl_rade', 1.0)
                features.append(pl_bmasse / (pl_rade ** 2))
            elif name == 'stellar_luminosity':
                st_rad = features_dict.get('st_rad', 1.0)
                st_teff = features_dict.get('st_teff', 5778)
                features.append((st_rad ** 2) * ((st_teff / 5778) ** 4))
            else:
                features.append(0.0)  # Default value
        
        # Scale and predict
        features_array = np.array([features])
        features_scaled = self.scaler.transform(features_array)
        prediction = self.model.predict(features_scaled, verbose=0)[0]
        
        # Create result dictionary
        result = {}
        for i, molecule in enumerate(self.target_molecules):
            result[molecule] = float(prediction[i])
        
        return result

if __name__ == "__main__":
    # Train the model
    model = ExoplanetAtmosphereModel()
    history = model.train(epochs=100)
    
    print("\nTraining complete!")
    print("Model files saved:")
    print("  - exoplanet_atmosphere_model.keras")
    print("  - scaler.pkl")
    print("  - feature_names.pkl")
    print("  - target_molecules.pkl")