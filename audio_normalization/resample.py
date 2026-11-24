import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import os
import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resample(input_path: str, output_path: str, target_rate: int = 16_000) -> None:
    """
    Resamples a .wav audio file to a specified sample rate.
    
    Parameters:
    - input_path (str): Path to the input .wav file.
    - output_path (str): Path where the resampled .wav file will be saved.
    - target_rate (int): Desired target sample rate (e.g., 16000, 22050, 44100).
    """
    audio_data, original_rate = sf.read(input_path)
    
    if original_rate == target_rate:
        print(f"Sample rate is already {target_rate} Hz. Copying file...")
        sf.write(output_path, audio_data, target_rate)
        return

    # Using that the GCD(original, target) yields that upscaling rate = target / GCD, and downsampling rate = original / GCD j
    gcd = np.gcd(original_rate, target_rate) # Greatest common denomitator between current and target rate
    up = target_rate // gcd # Rate at which we upsample
    down = original_rate // gcd  # rate at which we downsample the umpsampled audio

    resampled_data = resample_poly(audio_data, up, down, axis=0)

    sf.write(output_path, resampled_data, target_rate)


if __name__ == '__main__':
    # TODO: change this to point to a yaml config file to modularize
    path: os.PathLike = "/home/hdutheil/Documents/EPFL/ML/Audio_files/Audio Files/"
    output_dir: os.PathLike = "/home/hdutheil/Documents/EPFL/ML/Audio_files/resampled"

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(path):
        logger.info("Resampling directory")
        for file in os.scandir(path):
            file_name, extension = os.path.splitext(file.name)
            if file.is_file() and extension == '.wav':
                resample(file.path, os.path.join(output_dir, file.name), 16_000)

    elif os.path.isfile(path):
        *_, file_name, extension = file.path.split()
        if file.is_file() and extension == '.wav':
            logger.info("Resampling file")
            resample(path, os.path.join(output_dir, file_name), 16_000)
        else:
            logger.error("Not a .wav file, can't process it")
    else:
        logger.error("No such file or directory")
