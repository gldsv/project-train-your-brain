from train_your_brain.flow import build_flow
import argparse
import os

API_TOKEN = os.environ.get("API_TOKEN")
AZURE_TOKEN = os.environ.get("AZURE_TOKEN")
JEU_MILLE_EUROS_ID = "48f74c6a-eb98-11e1-a7b7-782bcb76618d_1" # Jeu des 1000 € unique ID
number_diffusions = "1" # Last diffusion

storage_dir = './raw_data'
# dir = os.environ.get("LOCAL_DATA_PATH")
filename = 'podcast_history.csv'
history_path = os.path.join(storage_dir, filename)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dev", help = "If specified, set dev mode for quicker flow", action = "store_true")
args = parser.parse_args()


if __name__ == "__main__":

    if args.dev:
        env = "dev"
    else:
        env = "prod"

    flow = build_flow(API_TOKEN, AZURE_TOKEN, JEU_MILLE_EUROS_ID, number_diffusions, env, storage_dir)
    flow.run()
