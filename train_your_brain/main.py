from train_your_brain.data import GetData
from train_your_brain.preproc_audio import Audio
from train_your_brain.retranscription import Retranscript
from train_your_brain.chunk_text import Chunk
import os

API_TOKEN = os.environ.get("API_TOKEN")
AZURE_TOKEN = os.environ.get("AZURE_TOKEN")
JEU_MILLE_EUROS_ID = "48f74c6a-eb98-11e1-a7b7-782bcb76618d_1" # Jeu des 1000 € unique ID
number_diffusions = "1" # Last diffusion

storage_dir = './raw_data'
# dir = os.environ.get("LOCAL_DATA_PATH")
filename = 'podcast_history.csv'
history_path = os.path.join(storage_dir, filename)

if __name__ == '__main__':

    print(f"⚙️ Getting last available episode")

    get_data = GetData(API_TOKEN)
    show_url = get_data.get_show_url(JEU_MILLE_EUROS_ID)
    diffusions_df = get_data.get_last_diffusions(show_url, number_diffusions)
    last_diffusion_url = get_data.get_last_diffusion_url(diffusions_df)
    last_diffusion_date = diffusions_df["date"].to_string(index=False)
    # get_data.save_diffusion_to_history(diffusions_df, history_path)

    print(f"⚙️ Preprocessing episode for {last_diffusion_date}")

    preprocessed_audio = Audio(last_diffusion_date, last_diffusion_url)
    audio_path = preprocessed_audio.export_conversion(storage_dir)

    print(f"✅ Cleaned and stored episode for {last_diffusion_date}")
    # print(audio_path)

    print(f"⚙️ Converting to text episode for {last_diffusion_date}")

    retranscript = Retranscript(AZURE_TOKEN)
    transcript_path = retranscript.speech_recognize_continuous_from_file(audio_path)

    print(f"✅ Converted to text episode for {last_diffusion_date}")

    print(f"⚙️ Chunking text episode for {last_diffusion_date}")

    with open(transcript_path) as f:
        text = f.readlines()
        text = text[0]

    chunker = Chunk(text)
    chunked_text = chunker.chunk_text()

    print(f"✅ Text chunked for episode for {last_diffusion_date}")
