import pandas as pd

from preprocess import Preprocess

COVID_DATA_PATH = '../data/basic/covid'


class CovidPreprocess(Preprocess):
    """
    The class responsible for preprocessing covid data.
    """

    def __init__(self, path: str = COVID_DATA_PATH) -> None:
        """
        Constructs all the necessary attributes for the **CovidPreprocess** object.
        :param path: path to dataset (*covid* default)
        """

        prepared_data = self.prepare_covid_data(path)
        super().__init__(prepared_data, path)
        super().preprocess()

    @staticmethod
    def prepare_covid_data(path: str = COVID_DATA_PATH) -> pd.DataFrame:
        """
        Prepares covid data for later preprocessing.
        :param path: path to dataset (*covid* default)
        :return: prepared covid data
        """

        df = pd.read_csv(path + '/covid.csv')
        df.rename(columns={"outcome": "label"}, inplace=True)

        return df
