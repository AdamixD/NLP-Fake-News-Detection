import pandas as pd
from preprocess import Preprocess

COVID_19 = '../data/basic/covid'
POLITICS = '../data/basic/political_and_news'
WORLD_NEWS = '../data/world'

class Politics(Preprocess):
    def __init__(self, path: str) -> None:
        self.path = path
        self.preprocess_politics()

    def preprocess_politics(self):
        true_data = pd.read_csv(self.path + '/True.csv')
        fake_data = pd.read_csv('/Fake.csv')
        true_data["label"] = 1
        fake_data["label"] = 0
        df = pd.concat([true_data, fake_data])
        df.drop(["title", "subject", "date"], axis=1, inplace=True)
        self.path_to_save = './processed/political_and_news'
        super().preprocess(df)


class WorldNews(Preprocess):
    def __init__(self, path: str) -> None:
        self.path = path
        self.preprocess_world_news()

    def preprocess_world_news(self):
        df = pd.read_csv(self.path + '/world_news.csv')
        df.drop(columns=['id', 'author'], inplace=True)
        df.text = df.title + df.text
        df.drop(columns=['title'], inplace=True)
        df.dropna(inplace=True)
        self.path_to_save = './processed/world_news'
        super().preprocess(df)


class Covid(Preprocess):
    def __init__(self, path: str) -> None:
        self.path = path
        self.preprocess_covid()

    def preprocess_covid(self):
        df = pd.read_csv(self.path + '/covid.csv')
        df.rename(columns={"outcome":"label"}, inplace=True)
        self.path_to_save = './processed/covid'
        super().preprocess(df)
