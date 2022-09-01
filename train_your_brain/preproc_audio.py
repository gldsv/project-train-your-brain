from pydub import AudioSegment
from copy import copy
import requests
import os

class Audio:
    """A class for audio processing."""

    def __init__(self, date: str, url: str, env: str):
        """Constructor for Audio class."""

        self.filename = date
        self.source_url = url
        #self.filepath = source
        self.audio = self.load_source(env)


    def load_source(self, env):
        """Load audio-source as AudioSegment."""

        audio_url = requests.get(self.source_url, allow_redirects=True)

        #get an audio file to work on
        with open(self.filename, 'wb') as f:
            f.write(audio_url.content)

        self.format = "mp3"
        self.audio = AudioSegment.from_file(self.filename)
        self.audio = self.audio.set_channels(1)
        self.audio = self.audio.set_frame_rate(16000)

        if env == "prod":
            self.audio = self.audio[180000:]
        if env == "dev":
            self.audio = self.audio[300000:360000]

        return self.audio


    def export_conversion(self, storage_dir):
        """Export a .wav sample from audio-source.
        Samples are kept in self.samples list.
        Inputs are in seconds.
        """

        # Create samples folder if doesn't exist
        storage_dir = storage_dir

        # Create output filepath
        output_filename = f'{os.path.splitext(os.path.basename(self.filename))[0]}.wav'
        output_path = os.path.join(storage_dir, output_filename)

        # Cancel if file already exists
        if os.path.exists(output_path):
            print(f'{os.path.basename(output_path)}: File already exists. Loading it.')
            return output_path

        # load_source
        if not self.audio:
            self.audio = self.load_source()

        # Generate output file
        try:
            self.audio.export(output_path, format='wav')
            os.remove(self.filename)
            print(f'{os.path.basename(output_path)}: File created')
            print('Sampling Succeeded')
        except Exception as e:
            print(f'Sampling Aborted : {e}')

        return output_path
