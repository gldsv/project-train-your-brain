import azure.cognitiveservices.speech as speechsdk
import time
import os

class Retranscript:
    """A class for converting audio to text"""

    def __init__(self, token):
        """Constructor for Azure API"""

        self.token = token


    def speech_recognize_continuous_from_file(self, audio_path:str, env:str):
        """performs continuous speech recognition with input from an audio file"""

        # <SpeechContinuousRecognitionWithFile>
        speech_config = speechsdk.SpeechConfig(subscription=self.token,region="westeurope",speech_recognition_language="fr-FR")
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        done = False

        def stop_cb(evt):
            """callback that stops continuous recognition upon receiving an event `evt`"""

            print('CLOSING on {}'.format(evt))
            speech_recognizer.stop_continuous_recognition()
            nonlocal done
            done = True

        all_results = []
        result = []

        def handle_final_result(evt):
            all_results.append(evt.result.text)

        speech_recognizer.recognized.connect(handle_final_result)
        # Connect callbacks to the events fired by the speech recognizer
        #speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
        #speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
        #speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        #speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        #speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

        # stop continuous recognition on either session stopped or canceled events
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        # Start continuous speech recognition
        speech_recognizer.start_continuous_recognition()

        while not done:
            time.sleep(.5)

        print("Printing all results:")
        #print(all_results)
        result = " ".join(all_results)

        transcript_path = f"{audio_path[:-4]}_{env}.txt"

        file = open(transcript_path,'w')
        file.write(result)

        os.remove(audio_path)

        return transcript_path
