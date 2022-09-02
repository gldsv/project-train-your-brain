from train_your_brain.data import GetData
from train_your_brain.preproc_audio import Audio
from train_your_brain.retranscription import Retranscript
from train_your_brain.chunk_text import Chunk
from train_your_brain.flow import run_flow
import os

API_TOKEN = os.environ.get("API_TOKEN")
AZURE_TOKEN = os.environ.get("AZURE_TOKEN")
JEU_MILLE_EUROS_ID = "48f74c6a-eb98-11e1-a7b7-782bcb76618d_1" # Jeu des 1000 € unique ID
number_diffusions = "1" # Last diffusion

storage_dir = './raw_data'
# dir = os.environ.get("LOCAL_DATA_PATH")
filename = 'podcast_history.csv'
history_path = os.path.join(storage_dir, filename)

env = "prod" # ["dev", "prod"]


if __name__ == "__main__":
    flow = run_flow(API_TOKEN, AZURE_TOKEN, JEU_MILLE_EUROS_ID, number_diffusions, env, storage_dir)
