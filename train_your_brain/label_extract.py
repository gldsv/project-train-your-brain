import os
import json
import pandas as pd

class LabelledData:
    """A class to extract labelled data from JSON, add labels for non labelled data and
        store it in dataframe with : Label, Text, characters posisition Start and End and Epidsode number """

    def __init__(self, url: str):
        """Constructor for LabelledData class."""
        #url of folder with labelled json files
        self.source_url = url

    def load_json(self):
        """Load json from Labelling Task."""

        #get list of all json in folder
        path_to_json =  self.source_url
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')& pos_json.startswith('from') ]

        #load all json and store json data in a list
        all_data_json = []
        for files in json_files :
            url = f'{path_to_json}/{files}'
            with open(url, 'r') as file:
                all_data_json.append(json.load(file))

        return all_data_json


    def extract_labels_from_json(self) :
        """Extract Labels and complete non labelled text """

        #Empty list for data storage and variable for number of episode QC
        labels_text = []
        nb_episode = 0
        all_raw_text = []

        # Get data from json with load_json function
        all_data_json = self.load_json()

        # Iterate over json data
        for json in all_data_json :

            data_several = json

            # One json file could contains several episode - Iteration to have each episode
            for episode in data_several :
                data_label = episode['label'] # All labeled data
                data_id = int(episode['id']) # Date of the episode
                raw_text = episode['text'] # Raw text for non labelled text extarction
                nb_episode += 1

                # Parameters
                nb_labels = len(data_label)
                first_label_start = data_label[0]['start']

                # If episode starts with non labelled data : Retrieve non labelled text and add "Blabla" label
                if first_label_start!= 0:
                    start = 0
                    end = first_label_start -1
                    labels_text.append(['Episode Start', '','','',data_id])
                    labels_text.append(['Blabla', raw_text[start:end],start,end,data_id])
                    n = first_label_start
                else :
                    n=0

                # Retrieve all the labels in the episode and store labels in list
                for i in range(0,nb_labels) :
                    # Index corresponding to a label part
                    if data_label[i]['start'] == n :
                        labels_text.append([data_label[i]['labels'][0], data_label[i]['text'],data_label[i]['start'],data_label[i]['end'],data_id])
                        n = data_label[i]['end'] +1

                    #Index not corresponding to a label part : Retrieve text from raw_text and add label "Blabla"
                    #Store next label
                    else :
                        len_max = data_label[i]['start']-1
                        labels_text.append(['Blabla', raw_text[n:len_max],n,len_max,data_id])
                        n=len_max +1
                        labels_text.append([data_label[i]['labels'][0], data_label[i]['text'],data_label[i]['start'],data_label[i]['end'],data_id])
                        n = data_label[i]['end'] +1

                # If episode ends with non labelled data : Retrieve non labelled text and add "Blabla" label
                if data_label[-1]['end'] < len(raw_text) :
                    start = data_label[-1]['end'] + 1
                    end = len(raw_text)
                    labels_text.append(['Blabla', raw_text[start:end],start,end,data_id])
                    labels_text.append(['Episode End', '','','',data_id])

                # Stcok raw Text
                all_raw_text.append([raw_text,data_id])

        #Transform list in DataFrame
        labels_text = pd.DataFrame(labels_text, columns=['label','text', 'label_start' , 'label_end', 'episode'] )



        return labels_text,  all_raw_text, nb_episode

    def preprocessed_label_to_json(self) :
        #Retriev pd.DataFrame
        df = self.extract_labels_from_json()
        date_start = df['episode'].min()
        date_end = df['episode'].max()
        #Extart pd.DataFrame to JSON
        df.to_json(f'{self.source_url}/label_preprocessed_export_from_{date_start}_to_{date_end}.json',force_ascii=False,orient='records')
