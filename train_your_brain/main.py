from train_your_brain.flow import build_flow
from datetime import datetime, timedelta
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


project_name = os.environ.get("PREFECT_PROJECT_NAME")


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dev", help = "If specified, set dev mode for quicker flow", action = "store_true")
args = parser.parse_args()


now = datetime.today()
now_hour = now.hour
SCHEDULE_HOUR = 14

if now_hour >= SCHEDULE_HOUR:
    date_to_process = now.strftime('%Y%m%d')
else:
    date_to_process = now - timedelta(1)
    date_to_process = date_to_process.strftime('%Y%m%d')


if __name__ == "__main__":

    if args.dev:
        env = "dev"
        print("🏗️ DEVELOPMENT MODE")
    else:
        env = "prod"
        print("🏢 PRODUCTION MODE")

    print(f"ℹ️ We're before {SCHEDULE_HOUR} h, processing for date {date_to_process}")
    flow = build_flow(date_to_process, API_TOKEN, AZURE_TOKEN, JEU_MILLE_EUROS_ID, number_diffusions, env, storage_dir)
    flow.run()
    # flow.register(project_name)
