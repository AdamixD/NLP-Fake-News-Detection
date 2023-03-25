import pandas as pd
from preprocess import Preprocess

COVID_19 = '../data/basic/covid'
POLITICS = '../data/basic/political_and_news'
WORLD_NEWS = '../data/basic/world'

class Politics(Preprocess):
    def __init__(self, path: str) -> None:
        self.clean_politics()
        super().__init__(self.df)
        super().preprocess(path)

    def clean_politics(self) -> None:
        true_data = pd.read_csv(self.path + '/True.csv')
        fake_data = pd.read_csv(self.path + '/Fake.csv')
        true_data["label"] = 1
        fake_data["label"] = 0
        df = pd.concat([true_data, fake_data])
        df.drop(["title", "subject", "date"], axis=1, inplace=True)
        self.df = df

class WorldNews(Preprocess):
    def __init__(self, path: str) -> None:
        self.clean_world_news()
        super().__init__(self.df)
        super().preprocess(path)

    def clean_world_news(self) -> None:
        df = pd.read_csv(self.path + '/world_news.csv')
        df.drop(columns=['id', 'author'], inplace=True)
        df.text = df.title + df.text
        df.drop(columns=['title'], inplace=True)
        df.dropna(inplace=True)
        self.df = df

class Covid(Preprocess):
    def __init__(self, path: str) -> None:
        self.clean_covid()
        super().__init__(self.df)
        super().preprocess(path)

    def clean_covid(self) -> None:
        df = pd.read_csv(self.path + '/covid.csv')
        df.rename(columns={"outcome":"label"}, inplace=True)
        self.df = df


