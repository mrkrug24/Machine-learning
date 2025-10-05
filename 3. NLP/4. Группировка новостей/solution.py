import re
import json
import nltk
import numpy as np
from pymystem3 import Mystem
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

class Solution:
    def __init__(self):
        nltk.download('stopwords')
        self.mystem = Mystem()
        self.t = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('russian'))
        data = json.load(open('dev-dataset-task2024-04.json'))
        self.x = self.t.fit_transform([self.text_processing(x) for x, _ in data]).toarray()
        self.y = np.array([int(y) for _, y in data])
        self.model = KNeighborsRegressor(n_neighbors=1, metric='cosine').fit(self.x, self.y)

    def text_processing(self, text: str) -> str:
        text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
        text = " ".join(self.mystem.lemmatize(text))
        return text

    def predict(self, text: str) -> str:
        self.x = np.vstack([self.x, v := self.t.transform([self.text_processing(text)]).toarray()])
        self.y = np.append(self.y, self.model.predict(v))
        self.model.fit(self.x, self.y)
        return str(int(self.y[-1]))