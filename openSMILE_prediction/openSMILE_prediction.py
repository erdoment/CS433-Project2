import opensmile
import yaml
from paths import CONFIG_PATH
import os
from typing import Any
import numpy as np
from tqdm import tqdm
import pandas as pd


def predict(file: os.PathLike) -> Any:
    if not os.path.isfile(file) or file.lower().endswith('.wav'):
        smile = opensmile.Smile(
                feature_set = opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors
                )
        return smile.process_file(file)
    return None

if __name__ == '__main__':
    with open(CONFIG_PATH) as config_path:
        config = yaml.safe_load(config_path)

    feature_path: os.PathLike = config['paths']['opensmile_feature_prediction_path']
    os.makedirs(feature_path, exist_ok=True)
        
    dir: os.PathLike = config['paths']['resampled_audio']

    valid_files = [
                file for file in os.scandir(dir) 
                if file.is_file() and file.name.lower().endswith('.wav')
            ]
    
    for file in tqdm(valid_files, desc='Extracting and saving file features'):
            file_base_name, _ = os.path.splitext(file.name)
            output_file_path = os.path.join(feature_path, f"{file_base_name}.npy")
            
            if not os.path.isfile(output_file_path):
                features: pd.DataFrame = predict(file.path)
                if features is not None:
                    np.save(output_file_path, features)
        
