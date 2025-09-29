"""
Data preprocessing module
Filter records with RL or EAB values and molecular formulas, parse chemical structures
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


class DataProcessor:
    def __init__(self, json_file_path: str):
        """Initialize data processor"""
        self.json_file_path = json_file_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> Dict:
        """Load JSON data"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        return self.raw_data
    
    def filter_valid_records(self) -> List[Dict]:
        """Filter records with RL or EAB values and molecular formulas"""
        if not self.raw_data:
            self.load_data()
            
        valid_records = []
        
        for paper in self.raw_data:
            if 'content' not in paper:
                continue
                
            for record in paper['content']:
                has_formula = self._check_chemical_formula(record)
                has_values = self._check_microwave_properties(record)
                
                if has_formula and has_values:
                    record['doi'] = paper.get('doi', '')
                    valid_records.append(record)
        
        print(f"Filtered {len(valid_records)} valid records")
        return valid_records
    
    def _check_chemical_formula(self, record: Dict) -> bool:
        """Check if record has chemical formula"""
        try:
            general_props = record.get('general_properties', {})
            formula = general_props.get('chemical_formula', '')
            return bool(formula and formula.strip())
        except:
            return False
    
    def _check_microwave_properties(self, record: Dict) -> bool:
        """Check if record has RL or EAB values"""
        try:
            mw_props = record.get('microwave_absorption_properties', {})
            
            rl_min = mw_props.get('rl_min', {})
            if isinstance(rl_min, dict) and 'value' in rl_min:
                rl_value = rl_min['value']
                if isinstance(rl_value, (int, float)) and not np.isnan(rl_value):
                    return True
            
            eab = mw_props.get('eab', {})
            if isinstance(eab, dict) and 'value' in eab:
                eab_value = eab['value']
                if isinstance(eab_value, (int, float)) and not np.isnan(eab_value):
                    return True
                    
            return False
        except:
            return False
    
    def parse_chemical_formula(self, formula: str) -> Dict:
        """Parse chemical formula, identify heterostructures and supported structures"""
        result = {
            'original_formula': formula,
            'is_heterostructure': False,
            'is_supported': False,
            'main_component': '',
            'components': []
        }
        
        formula = formula.strip()
        
        if '/' in formula:
            result['is_heterostructure'] = True
            components = [comp.strip() for comp in formula.split('/')]
            result['components'] = components
            result['main_component'] = components[0]
        elif '@' in formula:
            result['is_supported'] = True
            parts = formula.split('@')
            if len(parts) == 2:
                support, material = parts[0].strip(), parts[1].strip()
                result['components'] = [material, support]
                result['main_component'] = material
        else:
            separators = ['-', '_', '·', '•', '∶', ':']
            for sep in separators:
                if sep in formula:
                    components = [comp.strip() for comp in formula.split(sep)]
                    result['components'] = components
                    result['main_component'] = components[0]
                    break
            
            if not result['components']:
                result['main_component'] = formula
                result['components'] = [formula]
        
        return result
    
    def extract_features(self, record: Dict) -> Dict:
        """Extract features from a record"""
        features = {}
        
        general_props = record.get('general_properties', {})
        mw_props = record.get('microwave_absorption_properties', {})
        
        formula = general_props.get('chemical_formula', '')
        features['chemical_formula'] = formula
        features['formula'] = formula  # Add formula column for feature_enhancer compatibility
        
        parsed_formula = self.parse_chemical_formula(formula)
        features.update(parsed_formula)
        
        rl_min = mw_props.get('rl_min', {})
        if isinstance(rl_min, dict) and 'value' in rl_min:
            features['rl_min'] = rl_min['value']
        else:
            features['rl_min'] = np.nan
            
        eab = mw_props.get('eab', {})
        if isinstance(eab, dict) and 'value' in eab:
            features['eab'] = eab['value']
        else:
            features['eab'] = np.nan
        
        features['doi'] = record.get('doi', '')
        features['record_designation'] = record.get('record_designation', '')
        
        thickness = general_props.get('thickness', {})
        if isinstance(thickness, dict) and 'value' in thickness:
            features['thickness'] = thickness['value']
        else:
            features['thickness'] = np.nan
        
        return features
    
    def create_classification_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create classification targets for RL and EAB"""
        df = df.copy()
        
        # Convert rl_min and eab to numeric, handling string values
        df['rl_min'] = pd.to_numeric(df['rl_min'], errors='coerce')
        df['eab'] = pd.to_numeric(df['eab'], errors='coerce')
        
        # Remove rows with invalid numeric values
        initial_count = len(df)
        df = df.dropna(subset=['rl_min', 'eab'])
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"   Removed {initial_count - final_count} rows with invalid RL/EAB values")
        
        # Create classification targets only if we have valid data
        if len(df) > 0:
            df['rl_class'] = np.where(df['rl_min'] <= -50, 0, 1)
            df['eab_class'] = np.where(df['eab'] <= 4, 0, 1)
            
            print(f"RL classification distribution:")
            print(f"  Class 0 (RL ≤ -50): {(df['rl_class'] == 0).sum()}")
            print(f"  Class 1 (RL > -50): {(df['rl_class'] == 1).sum()}")
            
            print(f"EAB classification distribution:")
            print(f"  Class 0 (EAB ≤ 4): {(df['eab_class'] == 0).sum()}")
            print(f"  Class 1 (EAB > 4): {(df['eab_class'] == 1).sum()}")
        else:
            print("❌ No valid numeric data found for classification targets")
        
        return df
    
    def process_data(self) -> pd.DataFrame:
        """Main processing pipeline"""
        print("🔄 Starting data preprocessing...")
        
        valid_records = self.filter_valid_records()
        
        if not valid_records:
            print("❌ No valid records found")
            return pd.DataFrame()
        
        print("🔬 Extracting features...")
        features_list = []
        for record in valid_records:
            features = self.extract_features(record)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        print("🎯 Creating classification targets...")
        df = self.create_classification_targets(df)
        
        self.processed_data = df
        
        print(f"✅ Data preprocessing completed!")
        print(f"   Total records: {len(df)}")
        print(f"   Features: {df.columns.tolist()}")
        
        return df
    
    def get_formula_statistics(self) -> Dict:
        """Get statistics about chemical formulas"""
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        stats = {
            'total_formulas': len(df),
            'heterostructures': (df['is_heterostructure'] == True).sum(),
            'supported_structures': (df['is_supported'] == True).sum(),
            'simple_compounds': ((df['is_heterostructure'] == False) & 
                               (df['is_supported'] == False)).sum()
        }
        
        return stats 