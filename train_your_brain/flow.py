from train_your_brain.data import GetData
from train_your_brain.preproc_audio import Audio
from train_your_brain.retranscription import Retranscript
from train_your_brain.chunk_text import Chunk
from prefect import task, Flow


@task
def get_data(API_TOKEN, JEU_MILLE_EUROS_ID, number_diffusions):

    print(f"⚙️ Getting last available episode")

    get_data = GetData(API_TOKEN)
    show_url = get_data.get_show_url(JEU_MILLE_EUROS_ID)
    diffusions_df = get_data.get_last_diffusions(show_url, number_diffusions)
    last_diffusion_url = get_data.get_last_diffusion_url(diffusions_df)
    last_diffusion_date = diffusions_df["date"].to_string(index=False)
    # get_data.save_diffusion_to_history(diffusions_df, history_path)

    last_diffusion_info = {
        "url": last_diffusion_url,
        "date": last_diffusion_date
    }

    return last_diffusion_info


@task
def preprocess_audio(last_diffusion_date, last_diffusion_url, env, storage_dir):

    print(f"⚙️ Preprocessing episode for {last_diffusion_date}")

    preprocessed_audio = Audio(last_diffusion_date, last_diffusion_url, env)
    audio_path = preprocessed_audio.export_conversion(storage_dir)

    print(f"✅ Cleaned and stored episode for {last_diffusion_date}")

    return audio_path


@task
def transcript_audio(AZURE_TOKEN, last_diffusion_date, audio_path, env):

    print(f"⚙️ Converting to text episode for {last_diffusion_date}")

    retranscript = Retranscript(AZURE_TOKEN)
    transcript_path = retranscript.speech_recognize_continuous_from_file(audio_path, env)

    print(f"✅ Converted to text episode for {last_diffusion_date}")

    return transcript_path


@task
def chunk_transcript(last_diffusion_date, transcript_path):

    print(f"⚙️ Chunking text episode for {last_diffusion_date}")

    with open(transcript_path) as f:
        text = f.readlines()
        text = text[0]

    chunker = Chunk(text, last_diffusion_date)
    chunked_text_df = chunker.chunk_text()
    chunked_text_dict = chunker.chunk_dict()

    # print(chunked_text_dict)

    print(f"✅ Text chunked for episode for {last_diffusion_date}")

    return chunked_text_df, chunked_text_dict


def build_flow(API_TOKEN, AZURE_TOKEN, JEU_MILLE_EUROS_ID, number_diffusions, env, storage_dir):

    with Flow(name="my_test") as flow:
        last_diffusion_info = get_data(API_TOKEN, JEU_MILLE_EUROS_ID, number_diffusions)
        audio_path = preprocess_audio(last_diffusion_info["date"], last_diffusion_info["url"], env, storage_dir)
        transcript_path = transcript_audio(AZURE_TOKEN, last_diffusion_info["date"], audio_path, env)
        chunked_text = chunk_transcript(last_diffusion_info["date"], transcript_path)

    return flow
