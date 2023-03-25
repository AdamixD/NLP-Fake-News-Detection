import emoji
import pandas as pd
import re
import string
from textblob import TextBlob
from spylls.hunspell import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import *


class Preprocess:
    """
    Class for preprocessing data related to news
    """
    def __init__(self, df) -> None:
        """
        saves pandas.DataFrame as an atribute
        :param df: analized dataframe
        """
        self.df = df

    def preprocess(self, path: str) -> None:
        """
        most important method here, performs full dataframe preprocess
        changes occurences in 'text' column
        adds changes to dataframe in place
        saves to .json file
        """
        #self.links_preprocess()
        #self.references_preprocess()
        self.preprocess_hashtags()
        self.preprocess_emojis()
        self.remove_unicode_chars()
        self.preprocess_punctuation()
        self.spellcheck()
        self.analyze_sentiment()
        self.convert_to_lowercase()
        self.remove_stopwords()
        self.lemmatization()
        self.save_to_json(path)


    @staticmethod
    def __find_hashtags(row: pd.DataFrame) -> list:
        hashtags = re.findall(r"#\w+", row['text'])
        hashtags = [re.sub(r"^#", "", hashtag) for hashtag in hashtags]
        return hashtags
    
    @staticmethod
    def __remove_hashtags(row: pd.DataFrame) -> str:
        return re.sub(r"#", "", row['text'])

    def preprocess_hashtags(self) -> None:
        """
        method creates new column with list of all hashtags
        which occured in text and removes sign '#' from text
        """
        self.df['hashtags'] = self.df.apply(self.__find_hashtags, axis=1)
        self.df['text'] = self.df.apply(self.__remove_hashtags, axis=1)
    

    @staticmethod
    def __remove_punctuation(row: pd.DataFrame) -> str:
        """
        remove punctuation from text
        :param row: row from dataframe
        :return: text after removing punctuation
        """
        return "".join([char for char in row['text'] if char not in string.punctuation])

    def preprocess_punctuation(self) -> None:
        """
        method removes punctuation from whole dataset
        """
        self.df['text'] = self.df.apply(self.__remove_punctuation, axis=1)


    @staticmethod
    def __find_emojis(row: pd.DataFrame) -> list:
        return emoji.distinct_emoji_list(row['text'])

    @staticmethod
    def __interpret_emojis(row: pd.DataFrame) -> list:
        """
        method translates all emojis in 'emojis' column
        :param row: row from dataframe
        :return: list of emojis, but translated to text
        """
        return [emoji.demojize(emoji_item, delimiters=("", "")) for emoji_item in row['emojis']]

    @staticmethod
    def __remove_emojis(row: pd.DataFrame) -> str:
        """
        method removes all emojis from text
        :param row: row from dataframe
        :return: clean text without emojis
        """
        return emoji.replace_emoji(row['text'])

    def preprocess_emojis(self) -> None:
        self.df['emojis'] = self.df.apply(self.__find_emojis, axis=1)
        self.df['emojis'] = self.df.apply(self.__interpret_emojis, axis=1)
        self.df['text'] = self.df.apply(self.__remove_emojis, axis=1)


    @staticmethod
    def __define_sentiment(row: pd.DataFrame) -> str:
        """
        method translates sentiment od the text from numerical
        divides sentiment to: positive, neutral, negative
        :return: name of the sentiment
        """
        return 'negative' if row['polarity'] < 0 else 'positive' if row['polarity'] > 0 else 'neutral'

    def analyze_sentiment(self) -> None:
        """
        method creates new columns:
        * polarity - with the senimenty polarity of the text (numerical value)
        * subjectivity - with numerical subjectivity of the text
        * sentiment - string with translates and simmplifies polarity
        """
        sentiment_items = [TextBlob(text) for text in self.df['text'].tolist()]
        self.df['polarity'] = [text.sentiment.polarity for text in sentiment_items]
        self.df['subjectivity'] = [text.sentiment.subjectivity for text in sentiment_items]
        self.df['sentiment'] = self.df.apply(self.__define_sentiment, axis=1)


    @staticmethod
    def __spellcheck_on_row(row: pd.DataFrame) -> str:
        text_tokens = word_tokenize(row['text'])
        dictionary = Dictionary.from_files('en_US')
        new_text = []

        for word in text_tokens:
            if dictionary.lookup(word):
                new_text.append(word)
            else:
                try:
                    new_text.append(next(dictionary.suggest(word)))
                except StopIteration:
                    print(f"Very misspelled word occurred: {word}")
                    new_text.append(word)

        return ' '.join(new_text)

    def spellcheck(self) -> None:
        self.df['text'] = self.df.apply(
            lambda row: self.__spellcheck_on_row(row), axis=1
            )
        

    @staticmethod
    def __remove_unicode_chars_row(row: pd.DataFrame) -> str:
        return row['text'].encode("ascii", "ignore").decode()

    def remove_unicode_chars(self) -> None:
        self.df['text'] = self.df.apply(
            lambda row: self.__remove_unicode_chars_row(row), axis=1
            )
        

    def convert_to_lowercase(self) -> None:
        self.df['text'] = self.df['text'].apply(lambda text: text.lower())

    def save_to_json(self, path: str = '') -> None:
        """
        method saving dataframe to .json file
        if path is not specified, dataframe is saved in
        './processed/{dataframe_name}'
        """
        save_path = path_to_save(path)
        self.df.to_json(save_path, orient="records", lines=True)

    @staticmethod
    def __remove_stopwords_row(row: pd.DataFrame) -> str:
        """
        method performs removing stopwords on whole dataframe
        :param row: row from dataframe
        :return: text without stopwords
        """
        text_tokens = word_tokenize(row['text'])
        tokens_without_sw = [word for word in text_tokens
                             if word not in stopwords.words()]

        return ' '.join(tokens_without_sw)

    def remove_stopwords(self) -> None:
        """
        method performs removing stopwords on whole dataframe
        """
        self.df['text'] = self.df.apply(
            lambda row: self.__remove_stopwords_row(row), axis=1
            )
        
    @staticmethod
    def __lemmatize_text_row(row: pd.DataFrame) -> str:
        """
        method performs lemmatization on one row
        :param row: row from dataframe
        :return: text after lemmatization
        """
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(row['text'])
        new_text = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(new_text)

    def lemmatization(self) -> None:
        """
        method performs lemmatization on whole dataframe
        """
        self.df['text'] = self.df.apply(
            lambda row: self.__lemmatize_text_row(row), axis=1
            )
