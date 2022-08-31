from train_your_brain.data import GetData
from train_your_brain.preprop_audio import Audio
import os
import pandas as pd

API_TOKEN = os.environ.get("API_TOKEN")
JEU_MILLE_EUROS_ID = "48f74c6a-eb98-11e1-a7b7-782bcb76618d_1" # Jeu des 1000 € unique ID
number_diffusions = "1" # Last diffusion

dir = './raw_data'
# dir = os.environ.get("LOCAL_DATA_PATH")
filename = 'podcast_history.csv'
history_path = os.path.join(dir, filename)

if __name__ == '__main__':

    print(f"⬇️ Getting last available episode")

    get_data = GetData(API_TOKEN)
    show_url = get_data.get_show_url(JEU_MILLE_EUROS_ID)
    diffusions_df = get_data.get_last_diffusions(show_url, number_diffusions)
    last_diffusion_url = get_data.get_last_diffusion_url(diffusions_df)
    last_diffusion_date = diffusions_df["date"].to_string(index=False)
    # get_data.save_diffusion_to_history(diffusions_df, history_path)

    print(f"⚙️ Processing episode for {last_diffusion_date}")

    preprocessed_audio = Audio(last_diffusion_date, last_diffusion_url)
    preprocessed_audio.export_conversion()

    print(f"✅ Cleaned and stored episode for {last_diffusion_date}")
