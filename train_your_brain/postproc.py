import os
import numpy as np
import pandas as pd
from tokenizor_predict_data import Tokenizor_prediction
from IPython.display import HTML

class Postproc():

    def __init__(self, text, predict_data:Tokenizor_prediction, preds):
        #load the df from tokenizor_labelled_data.py
        self.df = predict_data
        self.df['episode_id']=self.df['episode_id'].apply(lambda x : x[0] if x else np.nan)
        self.df = self.df[~self.df["chunk_id"].isin([[0],[]])]
        #zoom episode 20220527
        self.df = self.df.query("episode_id==20220527")
        #load the predictions after the model
        self.preds = np.load(preds)
        self.txt = open(text).read()
        self.targets = np.array([np.array(i) for i in self.df['targets'].values])
        self.offset_mapping = np.array([np.array(i) for i in self.df['offset_mapping'].values])
        self.preds_flatten = self.preds.argmax(axis=-1)
        self.ab2idx={
            "B-Annonce_Question":1,
            "I-Annonce_Question":2,
            "B-Question":3,
            "I-Question":4,
            "B-Bonne_Reponse":5,
            "I-Bonne_Reponse":6,
            "B-Confirmation_Reponse":6,
            "I-Confirmation_Reponse":6,
            "B-Mauvaise_Reponse":7,
            "I-Mauvaise_Reponse":8,
            "B-To_delete":0,
            "I-To_delete":0,
            "B-Blabla":0,
            "I-Blabla":0,
            }
        self.idx2lab = {v:k[2:] for k,v in self.ab2idx.items()if v > 0}

    def predictator(self,y_pred,offset_mapping_in):
        if type(y_pred[0])==list:
            y_pred=np.array([np.array(i) for i in y_pred])
        if len(y_pred.shape)>=2:
            y_pred = y_pred.argmax(axis=-1)

        labels=[]
        offset_mapping=[]

        for l,om in zip(y_pred,offset_mapping_in):
            if om[0]==0 and om[1]==0:
                continue
            if len(labels) < 1 :
                labels.append(l)
                offset_mapping.append(np.array(om))
            elif labels[-1]!=l:
                labels.append(l)
                offset_mapping.append(np.array(om))
            else:
                new_om = np.array([offset_mapping[-1][0],om[1]])
                del offset_mapping[-1]
                offset_mapping.append(new_om)

        return labels,offset_mapping

    def aggregation_predicator(self):
        self.labels_pred=[]
        self.offsetmapping_pred=[]

        for p,om in zip(self.preds,self.offset_mapping):
            labs_,om_=self.predictator(p,om)
            self.labels_pred.append(labs_)
            self.offsetmapping_pred.append(om_)

        self.labels_true=[]
        self.offsetmapping_true=[]

        for p,om in zip(np.array([np.array(i) for i in self.df.targets]),self.offset_mapping):
            labs,om_=self.predictator(p,om)
            self.labels_true.append(labs)
            self.offsetmapping_true.append(om_)

        return self


    def prettyfier_chunk(self,preds_agg,om_agg,txt):
        res = dict(labels=[],offset=[], html=[])
        for preds,oms in zip(preds_agg,om_agg):
            for p,om in zip(preds,oms):
                label = self.idx2lab.get(p,)
                txt_=txt[om[0]:om[1]]
                res['labels'].append(label)
                res['offset'].append((om[0],om[1]))
                res['html'].append("<{0} style='padding: 2px'>{1}</{0}>".format(label,txt_))
        return res

    def css():
        styles = open("/Users/laurene.merite/code/laurene-merite-wagon/project-train-your-brain/custom.css", "r").read()
        return HTML('<style>'+styles+'</style>')
    css()

    def comparison_text(self):
        respred = self.prettyfier_chunk(self.labels_pred,self.offsetmapping_pred,self.txt)
        restrue = self.prettyfier_chunk(self.labels_true,self.offsetmapping_true,self.txt)
        prediction = ' '.join(respred['html'])
        ground_truth = ' '.join(restrue['html'])
        html = f"""
        <div class="content">
        <span style="font-size:16px">Legend --></span>
        <Annonce_Question>Annonce_Question</Annonce_Question>
        <Question>Question</Question>
        <Bonne_Reponse>Bonne_Reponse</Bonne_Reponse>
        <Confirmation_Reponse>Confirmation_Reponse</Confirmation_Reponse>
        <Mauvaise_Reponse>Mauvaise_Reponse</Mauvaise_Reponse>
        </div>

        <div class="row">
        <div class="column">
            <h2 class="title">Prediction</h2>
            <p style="text-align:justify">{prediction}</p>
        </div>
        <div class="column">
            <h2 class="title">Ground Truth</h2>
            <p style="text-align:justify">{ground_truth}</p>
        </div>
        </div>

        """

        return HTML(html)

    def postprepoc_df(self):
        res_df = pd.DataFrame(self.prettyfier_chunk(self.labels_pred,self.offsetmapping_pred,self.txt))
        labels_=[]
        offsetmapping_=[]
        text_ = []

        for l,om in zip(res_df['labels'],res_df['offset']):
            if om[0]==0 and om[1]==0:
                    continue
            if len(labels_) < 1 :
                labels_.append(l)
                offsetmapping_.append(np.array(om))
                text_.append(self.txt[om[0]:om[1]])

            if labels_[-1]!=l:
                if l == None :
                    new_om = np.array([offsetmapping_[-1][0],om[1]])
                    del text_[-1]
                    text_.append(self.txt[new_om[0]:new_om[1]])
                    continue
                labels_.append(l)
                offsetmapping_.append(np.array(om))
                text_.append(self.txt[om[0]:om[1]])
            else:
                new_om = np.array([offsetmapping_[-1][0],om[1]])
                del offsetmapping_[-1]
                del text_[-1]
                offsetmapping_.append(new_om)
                text_.append(self.txt[new_om[0]:new_om[1]])
        tmp = pd.DataFrame({'labels' : labels_, 'offset' : offsetmapping_, 'text' : text_})
        return tmp


if __name__ == "__main__":
    url = os.path.join(f'./exchange')
    text = '/Users/laurene.merite/code/laurene-merite-wagon/project-train-your-brain/test/transcript_20220527.txt'
    predict_data = pd.read_json('/Users/laurene.merite/code/laurene-merite-wagon/project-train-your-brain/exchange/train_data/label_train_data_tokenized.json',orient='records',)
    preds = '/Users/laurene.merite/code/laurene-merite-wagon/project-train-your-brain/test/test7.npy'
    tmp = Postproc(text=text, predict_data=predict_data,preds=preds)
    tmp = tmp.aggregation_predicator()
    tmp = tmp.postprepoc_df()
    tmp.to_csv(f"{url}/postprocessed_data.csv")
    print(f"✅ Postprocessed episode")
