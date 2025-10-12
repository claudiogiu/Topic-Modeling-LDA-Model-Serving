from dataclasses import dataclass, field
import os
import json
import pandas as pd
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
from load_data import DataLoader
import warnings

warnings.filterwarnings("ignore")

@dataclass
class PreProcessor:
    """
    Interface for preprocessing textual datasets and generating tokenized documents and corpus artifacts for topic modeling.

    Attributes:
        text_column (str): Name of the column containing raw text to be processed.
        model_dir (str): Directory where the dictionary object is stored.
        processed_dir (str): Directory where preprocessed datasets and corpus files are saved.
        stop_words (set): Set of stopwords used for token filtering.
        lemmatizer (WordNetLemmatizer): Lemmatization engine applied to tokens.
        dictionary (corpora.Dictionary): Dictionary built from training data.

    Methods:
        fit_transform(df: pd.DataFrame, dataset_name: str) -> None:
            Applies preprocessing, builds dictionary, and saves both tokenized documents and BoW corpus.

        transform(df: pd.DataFrame, dataset_name: str) -> None:
            Applies preprocessing, generates corpus using a previously saved dictionary, and saves tokenized documents and BoW corpus.

        _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
            Tokenizes, filters, and lemmatizes text column to produce lists of lemmatized tokens.

        _pre_process(text: str) -> list[str]:
            Processes a single document into a list of cleaned, lemmatized tokens.

        _save_preprocessed(df: pd.DataFrame, dataset_name: str) -> None:
            Saves the preprocessed DataFrame to disk as CSV.

        _build_dictionary(texts: pd.Series) -> None:
            Constructs and saves a dictionary from tokenized training data.

        _load_dictionary() -> None:
            Loads a previously saved dictionary from disk.

        _build_and_save_corpus(texts: pd.Series, filename: str) -> None:
            Converts tokenized data into BoW corpus format and saves it as JSON.

        _basename(path: str) -> str:
            Extracts the base filename without extension.
    """

    text_column: str
    model_dir: str = "models"
    processed_dir: str = "data/processed"
    stop_words: set = field(default_factory=lambda: set(stopwords.words("english")))
    lemmatizer: WordNetLemmatizer = field(default_factory=WordNetLemmatizer)
    dictionary: corpora.Dictionary = field(default=None)

    def fit_transform(self, df: pd.DataFrame, dataset_name: str) -> None:
        df_clean = self._apply_preprocessing(df)
        self._save_preprocessed(df_clean, dataset_name)
        self._build_dictionary(df_clean["texts"])
        self._build_and_save_corpus(df_clean["texts"], f"{self._basename(dataset_name)}_corpus.json")

    def transform(self, df: pd.DataFrame, dataset_name: str) -> None:
        df_clean = self._apply_preprocessing(df)
        self._save_preprocessed(df_clean, dataset_name)
        self._load_dictionary()
        self._build_and_save_corpus(df_clean["texts"], f"{self._basename(dataset_name)}_corpus.json")

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_column not in df.columns:
            raise ValueError(f"Missing column '{self.text_column}' in input dataset.")
        df["texts"] = df[self.text_column].apply(self._pre_process)
        if df["texts"].apply(len).sum() == 0:
            raise ValueError("Preprocessing failed: all documents are empty.")
        return df[[self.text_column, "texts"]]

    def _pre_process(self, text: str) -> list[str]:
        tokens = simple_preprocess(text, deacc=True, min_len=3)
        return [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

    def _save_preprocessed(self, df: pd.DataFrame, dataset_name: str) -> None:
        filename = self._basename(dataset_name) + ".csv"
        path = os.path.join(self.processed_dir, filename)
        os.makedirs(self.processed_dir, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Preprocessed dataset '{filename}' successfully saved in the '{self.processed_dir}' folder.")

    def _build_dictionary(self, texts: pd.Series) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, "dictionary.dict")
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=150, no_above=0.1)
        self.dictionary.save(path)
        print(f"Dictionary file 'dictionary.dict' successfully saved in the '{self.model_dir}' folder.")
        print(f"Vocabulary of size: {len(self.dictionary):,} tokens.")

    def _load_dictionary(self) -> None:
        path = os.path.join(self.model_dir, "dictionary.dict")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary file not found at: {path}")
        self.dictionary = corpora.Dictionary.load(path)

    def _build_and_save_corpus(self, texts: pd.Series, filename: str) -> None:
        path = os.path.join(self.processed_dir, filename)
        corpus = [self.dictionary.doc2bow(doc) for doc in texts]
        if all(len(doc) == 0 for doc in corpus):
            raise ValueError("Corpus generation failed: no valid tokens found.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2)
        print(f"Corpus file '{filename}' successfully saved in the '{self.processed_dir}' folder.")

    def _basename(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    processor: PreProcessor = PreProcessor(text_column="ABSTRACT")

    train_loader: DataLoader = DataLoader(data_type="raw", dataset_name="train.csv")
    train_df: pd.DataFrame = train_loader.load()
    processor.fit_transform(train_df, "train.csv")

    test_loader: DataLoader = DataLoader(data_type="raw", dataset_name="test.csv")
    test_df: pd.DataFrame = test_loader.load()
    processor.transform(test_df, "test.csv")