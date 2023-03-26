import pandas as pd

from preprocess.preprocess import Preprocess

WORLD_NEWS_DATA_PATH = "data/basic/world_news"


class WorldNewsPreprocess(Preprocess):
    """
    The class responsible for preprocessing world news data.
    """

    def __init__(self, path: str = WORLD_NEWS_DATA_PATH) -> None:
        """
        Constructs all the necessary attributes for the **WorldNewsPreprocess** class object.
        :param path: path to dataset (*world_news* default)
        """

        prepared_data = self.prepare_world_news_data(path)
        super().__init__(prepared_data, path)

    @staticmethod
    def prepare_world_news_data(path: str = WORLD_NEWS_DATA_PATH) -> pd.DataFrame:
        """
        Prepares world news data for later preprocessing.
        :param path: path to dataset (*world_news* default)
        :return: prepared world news data
        """

        df = pd.read_csv(path + "/world_news.csv")
        df.drop(columns=['id', 'author'], inplace=True)
        df.text = df.title + df.text
        df.drop(columns=['title'], inplace=True)
        df.dropna(inplace=True)

        return df
