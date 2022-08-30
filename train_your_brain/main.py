from train_your_brain.data import GetData
import os

API_TOKEN = os.environ.get("API_TOKEN")
JEU_MILLE_EUROS_ID = "48f74c6a-eb98-11e1-a7b7-782bcb76618d_1" # Jeu des 1000 € unique ID
number_diffusions = "1" # Last diffusion

dir = './raw_data'
# dir = os.environ.get("LOCAL_DATA_PATH")
filename = 'podcast_history.csv'
history_path = os.path.join(dir, filename)

if __name__ == '__main__':
    get_data = GetData(API_TOKEN)
    show_url = get_data.get_show_url(JEU_MILLE_EUROS_ID)
    diffusions_df = get_data.get_last_diffusions(show_url, number_diffusions)
    get_data.save_diffusion_to_history(diffusions_df, history_path)
