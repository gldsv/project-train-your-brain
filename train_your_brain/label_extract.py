import os
import json
import pandas as pd

class LabelledData:
    """A class for extract labelled data from JSON, add labels for non labelled data and
        store in dataframe Label, Text, characters posisition start and end and Epidsode number """

    def __init__(self, url: str):
        """Constructor for LabelledData class."""
        self.source_url = url

    def load_json(self):
        """Load json from Labelling Task."""

        #get json list
        path_to_json =  self.source_url
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

        #load json and store data in list all_data_json
        all_data_json = []
        for files in json_files :
            url = f'{path_to_json}/{files}'
            with open(url, 'r') as file:
                all_data_json.append(json.load(file))

        return all_data_json


    def extract_labels_from_json(self) :
        """Extract Labels and complete non labelled text """

        ## Extract several episode with Blabla
        labels_text = []
        nb_episode = 0

        all_data_json = self.load_json()

        for json in all_data_json :

            data_several = json

            for episode in data_several :
                data_label = episode['label']
                data_id = episode['id']
                raw_text = episode['text']
                nb_episode += 1

                # Parameters
                len_raw_text = len(raw_text)
                nb_labels = len(data_label)
                first_label_start = data_label[0]['start']
                last_label_end = data_label[-1]['end']


                if first_label_start!= 0:
                    start = 0
                    end = first_label_start -1
                    labels_text.append(['Episode Start', '-','-','-',data_id])
                    labels_text.append(['Blabla', raw_text[start:end],start,end,data_id])
                    n = first_label_start
                else :
                    n=0

                for i in range(0,nb_labels) :
                    if data_label[i]['start'] == n :
                        labels_text.append([data_label[i]['labels'][0], data_label[i]['text'],data_label[i]['start'],data_label[i]['end'],data_id])
                        n = data_label[i]['end'] +1
                    else :
                        len_max = data_label[i]['start']-1
                        labels_text.append(['Blabla', raw_text[n:len_max],n,len_max,data_id])
                        n=len_max +1
                        labels_text.append([data_label[i]['labels'][0], data_label[i]['text'],data_label[i]['start'],data_label[i]['end'],data_id])
                        n = data_label[i]['end'] +1

                if data_label[-1]['end'] < len(raw_text) :
                    start = data_label[-1]['end'] + 1
                    end = len(raw_text)
                    labels_text.append(['Blabla', raw_text[start:end],start,end,data_id])
                    labels_text.append(['Episode End', '-','-','-',data_id])



        labels_text = pd.DataFrame(labels_text, columns=['Label','Text', 'Label_Start' , 'Label_End', 'Episode'] )


        return labels_text
