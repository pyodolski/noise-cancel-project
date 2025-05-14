from utils.audio_utils import create_dataset

create_dataset(
    clean_dir='data/clean_train',
    noise_dir='data/noise_train',
    out_dir='data/mixtures_train',
    snr_db=5
)
