"""
Exoplanet Data Collection from NASA Archive
Fetches and preprocesses data for atmospheric composition prediction
"""

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json

class ExoplanetDataCollector:
    def __init__(self):
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        self.scaler = StandardScaler()
        
    def fetch_exoplanet_data(self, limit=5000):
        """Fetch exoplanet data from NASA archive"""
        
        # Query to get exoplanet data with atmospheric parameters
        query = f"""
        SELECT TOP {limit}
            pl_name,
            pl_orbper,
            pl_rade,
            pl_bmasse,
            pl_orbeccen,
            st_teff,
            st_rad,
            st_mass,
            sy_dist,
            pl_eqt,
            pl_dens,
            st_logg,
            st_met,
            pl_orbsmax,
            pl_insol
        FROM ps 
        WHERE default_flag = 1
        AND pl_rade IS NOT NULL
        AND pl_orbper IS NOT NULL
        AND st_teff IS NOT NULL
        AND pl_eqt IS NOT NULL
        """
        
        params = {
            'query': query,
            'format': 'json'
        }
        
        print("Fetching data from NASA Exoplanet Archive...")
        response = requests.get(self.base_url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            print(f"Successfully fetched {len(df)} exoplanets")
            return df
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")
    
    def create_synthetic_atmospheric_labels(self, df):
        """
        Create synthetic atmospheric composition labels based on physical parameters
        This simulates what real spectroscopic data would provide
        
        Based on astrophysical principles:
        - Hot Jupiters (high temp, large radius): H2, He dominated
        - Super-Earths (moderate temp): H2O, CO2, N2
        - Cold planets: CH4, NH3, H2O ice
        """
        
        compositions = []
        
        for _, row in df.iterrows():
            comp = {}
            
            radius = row['pl_rade'] if pd.notna(row['pl_rade']) else 1.0
            temp = row['pl_eqt'] if pd.notna(row['pl_eqt']) else 300
            mass = row['pl_bmasse'] if pd.notna(row['pl_bmasse']) else 1.0
            
            # Hot Jupiter-like (T > 1000K, R > 8 Earth radii)
            if temp > 1000 and radius > 8:
                comp = {
                    'H2': np.random.uniform(75, 85),
                    'He': np.random.uniform(10, 20),
                    'H2O': np.random.uniform(0.1, 2),
                    'CO': np.random.uniform(0.1, 1),
                    'CH4': np.random.uniform(0.01, 0.5),
                    'CO2': np.random.uniform(0.01, 0.3),
                    'NH3': np.random.uniform(0.001, 0.1),
                    'N2': np.random.uniform(0.1, 1)
                }
            
            # Hot Neptune (T > 700K, 3 < R < 8)
            elif temp > 700 and 3 < radius < 8:
                comp = {
                    'H2': np.random.uniform(65, 75),
                    'He': np.random.uniform(15, 25),
                    'H2O': np.random.uniform(2, 8),
                    'CO': np.random.uniform(0.5, 2),
                    'CH4': np.random.uniform(0.5, 3),
                    'CO2': np.random.uniform(0.3, 1.5),
                    'NH3': np.random.uniform(0.1, 0.5),
                    'N2': np.random.uniform(0.5, 2)
                }
            
            # Warm Super-Earth (400 < T < 700, R < 3)
            elif 400 < temp < 700 and radius < 3:
                comp = {
                    'H2': np.random.uniform(20, 40),
                    'He': np.random.uniform(5, 15),
                    'H2O': np.random.uniform(15, 30),
                    'CO': np.random.uniform(1, 3),
                    'CH4': np.random.uniform(2, 8),
                    'CO2': np.random.uniform(10, 25),
                    'NH3': np.random.uniform(0.5, 2),
                    'N2': np.random.uniform(10, 20)
                }
            
            # Temperate planets (200 < T < 400)
            elif 200 < temp < 400:
                comp = {
                    'H2': np.random.uniform(10, 30),
                    'He': np.random.uniform(3, 10),
                    'H2O': np.random.uniform(20, 40),
                    'CO': np.random.uniform(0.5, 2),
                    'CH4': np.random.uniform(5, 15),
                    'CO2': np.random.uniform(15, 30),
                    'NH3': np.random.uniform(1, 5),
                    'N2': np.random.uniform(15, 25)
                }
            
            # Cold planets (T < 200)
            else:
                comp = {
                    'H2': np.random.uniform(50, 70),
                    'He': np.random.uniform(10, 20),
                    'H2O': np.random.uniform(0.5, 3),
                    'CO': np.random.uniform(0.1, 0.5),
                    'CH4': np.random.uniform(8, 20),
                    'CO2': np.random.uniform(0.5, 2),
                    'NH3': np.random.uniform(2, 8),
                    'N2': np.random.uniform(1, 5)
                }
            
            # Normalize to 100%
            total = sum(comp.values())
            comp = {k: (v/total)*100 for k, v in comp.items()}
            
            compositions.append(comp)
        
        return pd.DataFrame(compositions)
    
    def preprocess_data(self, df):
        """Preprocess and clean the data"""
        
        # Features to use
        feature_cols = [
            'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_orbeccen',
            'st_teff', 'st_rad', 'st_mass', 'sy_dist', 'pl_eqt'
        ]
        
        # Fill missing values with median
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Add derived features
        df['insolation_flux'] = (df['st_rad'] ** 2) / (df['pl_orbper'] ** (2/3))
        df['surface_gravity'] = df['pl_bmasse'] / (df['pl_rade'] ** 2)
        df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
        
        feature_cols.extend(['insolation_flux', 'surface_gravity', 'stellar_luminosity'])
        
        # Remove any remaining NaN rows
        df = df.dropna(subset=feature_cols)
        
        return df, feature_cols
    
    def prepare_dataset(self, save_path='exoplanet_dataset.csv'):
        """Complete pipeline to prepare dataset"""
        
        # Fetch data
        df = self.fetch_exoplanet_data()
        
        # Preprocess
        df, feature_cols = self.preprocess_data(df)
        
        # Create synthetic labels
        compositions = self.create_synthetic_atmospheric_labels(df)
        
        # Combine features and labels
        full_dataset = pd.concat([df[['pl_name'] + feature_cols].reset_index(drop=True), 
                                  compositions.reset_index(drop=True)], axis=1)
        
        # Save dataset
        full_dataset.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
        print(f"Total samples: {len(full_dataset)}")
        print(f"Features: {feature_cols}")
        print(f"Target molecules: {list(compositions.columns)}")
        
        # Save feature names and scaler
        joblib.dump(feature_cols, 'feature_names.pkl')
        
        return full_dataset, feature_cols

if __name__ == "__main__":
    collector = ExoplanetDataCollector()
    dataset, features = collector.prepare_dataset()
    print("\nDataset preparation complete!")
    print("\nFirst few rows:")
    print(dataset.head())