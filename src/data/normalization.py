"""
Feature normalisering med MinMaxScaler för neural network kompatibilitet.

Stödjer feature_range, NaN, konstanta kolumner, icke-numeriska kolumner och returnerar scalers vid behov.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, Optional

def normalize_features(df: pd.DataFrame, feature_range: Tuple[float, float] = (-1, 1), return_scalers: bool = False, scalers: Optional[Dict[str, Any]] = None) -> Any:
    """
    Normalisera numeriska features i en DataFrame till angivet intervall (default: [-1, 1]).
    Bevarar Time- och icke-numeriska kolumner oförändrade.
    Hanterar NaN och konstanta kolumner.
    
    Args:
        df: DataFrame med features
        feature_range: Tuple (min, max) för normalisering
        return_scalers: Om True, returnera även scalers per kolumn
        scalers: Befintliga scalers att använda (för data leakage förebyggande)
    Returns:
        Normaliserad DataFrame (och dict med scalers om return_scalers=True)
    """
    if df.empty:
        if return_scalers:
            return df.copy(), {}
        return df.copy()
    
    result = df.copy()
    new_scalers = {}
    
    # Identifiera numeriska kolumner (exkludera Time och icke-numeriska)
    exclude_cols = []
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]) or pd.api.types.is_timedelta64_dtype(result[col]):
            exclude_cols.append(col)
        elif pd.api.types.is_object_dtype(result[col]) or pd.api.types.is_bool_dtype(result[col]):
            exclude_cols.append(col)
    
    numeric_cols = [col for col in result.columns if col not in exclude_cols]
    
    for col in numeric_cols:
        col_data = result[col]
        
        # Hantera kolumner med endast NaN
        if col_data.isna().all():
            continue
            
        # Hantera kolumner med endast ett unikt värde (konstant)
        if col_data.nunique(dropna=True) == 1:
            if scalers and col in scalers:
                # Använd befintlig scaler för konstant kolumn
                scaler = scalers[col]
            else:
                # Skapa ny scaler för konstant kolumn
                scaler = MinMaxScaler(feature_range=feature_range)
                # Fyll NaN temporärt för att passa fit_transform
                temp = col_data.fillna(0).values.reshape(-1, 1)
                scaler.fit(temp)
            
            # Transform med scaler
            temp = col_data.fillna(0).values.reshape(-1, 1)
            scaled = scaler.transform(temp)
            # Återställ NaN
            scaled = pd.Series(scaled.flatten(), index=col_data.index)
            scaled[col_data.isna()] = np.nan
            result[col] = scaled
            new_scalers[col] = scaler
            continue
            
        # Hantera vanliga numeriska kolumner
        if scalers and col in scalers:
            # Använd befintlig scaler (för data leakage förebyggande)
            scaler = scalers[col]
            # Behåll NaN under transform
            mask = ~col_data.isna()
            values = col_data.values.reshape(-1, 1)
            scaled = np.full_like(values, np.nan, dtype=np.float64)
            scaled[mask] = scaler.transform(values[mask])
            result[col] = scaled.flatten()
            new_scalers[col] = scaler
        else:
            # Skapa ny scaler och fit på data
            scaler = MinMaxScaler(feature_range=feature_range)
            # Behåll NaN under transform
            mask = ~col_data.isna()
            values = col_data.values.reshape(-1, 1)
            scaled = np.full_like(values, np.nan, dtype=np.float64)
            scaled[mask] = scaler.fit_transform(values[mask])
            result[col] = scaled.flatten()
            new_scalers[col] = scaler
    
    if return_scalers:
        return result, new_scalers
    return result 