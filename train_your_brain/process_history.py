import os
import pandas as pd
from train_your_brain.preproc_audio import Audio
from train_your_brain.retranscription import Retranscript


AZURE_TOKEN = os.environ.get("AZURE_TOKEN")
storage_dir = "./exchange"
history_path = "./raw_data/podcast_history_2.csv"
history = pd.read_csv(history_path)
history_range = history["date"].values.tolist()


if __name__ == "__main__":
    for date in history_range:
        url = history["url"][history["date"] == date].tolist()[0]

        print(f"⚙️ Preprocessing episode for {date}")

        preprocessed_audio = Audio(str(date), url, "prod")
        audio_path = preprocessed_audio.export_conversion(storage_dir)

        print(f"✅ Cleaned and stored episode for {date}")

        print(f"⚙️ Converting to text episode for {date}")

        retranscript = Retranscript(AZURE_TOKEN)
        transcript_path = retranscript.speech_recognize_continuous_from_file(audio_path)

        print(f"✅ Converted to text episode for {date}")
