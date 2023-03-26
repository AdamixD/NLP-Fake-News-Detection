import pandas as pd

from preprocess.preprocess import Preprocess

POLITICAL_AND_NEWS_DATA_PATH = "data/basic/political_and_news"


class PoliticsPreprocess(Preprocess):
    """
    The class responsible for preprocessing political and news data.
    """

    def __init__(self, path: str = POLITICAL_AND_NEWS_DATA_PATH) -> None:
        """
        Constructs all the necessary attributes for the **PoliticsPreprocess** class object.
        :param path: path to dataset (*political_and_news* default)
        """

        prepared_data = self.prepare_politics_data(path)
        super().__init__(prepared_data, path)

    @staticmethod
    def prepare_politics_data(path: str) -> pd.DataFrame:
        """
        Prepares political and news data for later preprocessing.
        :param path: path to dataset (*political_and_news* default)
        :return: prepared political and news data
        """

        true_data = pd.read_csv(path + "/True.csv")
        fake_data = pd.read_csv(path + "/Fake.csv")
        true_data["label"] = 1
        fake_data["label"] = 0
        df = pd.concat([true_data, fake_data])
        df.text = df.title + df.text
        df.drop(["title", "subject", "date"], axis=1, inplace=True)

        return df
