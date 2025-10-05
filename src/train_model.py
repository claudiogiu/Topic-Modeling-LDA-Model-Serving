from dataclasses import dataclass, field
import os
import json
from gensim import corpora
from gensim.models import LdaModel

import warnings

warnings.filterwarnings("ignore")

@dataclass
class ModelTrainer:
    """
    Interface for training a LDA model using preprocessed corpus and dictionary artifacts.

    Attributes:
        corpus_filename (str): Name of the corpus file located in the processed data directory.
        dictionary_filename (str): Name of the dictionary file located in the model artifact directory.
        model_filename (str): Name of the output file where the trained LDA model is saved.
        num_topics (int): Number of latent topics to extract from the corpus.
        passes (int): Number of full passes over the corpus during training.
        alpha (str | float | list | None): Dirichlet prior for document-topic distribution.
        eta (str | float | list | None): Dirichlet prior for topic-word distribution.
        model_dir (str): Directory where model artifacts are stored.
        processed_dir (str): Directory containing preprocessed corpus files.

    Methods:
        train() -> None:
            Loads dictionary and corpus, trains the LDA model, and saves it to disk.

        _load_dictionary() -> None:
            Loads a previously saved dictionary from disk.

        _load_corpus() -> None:
            Loads a preprocessed corpus from disk in JSON format.

        _save_model() -> None:
            Persists the trained LDA model to disk.
    """

    corpus_filename: str = "train_corpus.json"
    dictionary_filename: str = "dictionary.dict"
    model_filename: str = "lda_model.gensim"
    num_topics: int = 10
    passes: int = 20
    alpha: str | float | list | None = "auto"
    eta: str | float | list | None = "auto"
    model_dir: str = "models"
    processed_dir: str = "data/processed"

    dictionary: corpora.Dictionary = field(init=False)
    corpus: list = field(init=False)
    lda_model: LdaModel = field(init=False)

    def train(self) -> None:
        self._load_dictionary()
        self._load_corpus()

        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            alpha=self.alpha,
            eta=self.eta,
            random_state=42
        )

        self._save_model()
        
        print(f"LDA model trained and successfully saved in '{self.model_dir}' folder.")

    def _load_dictionary(self) -> None:
        path = os.path.join(self.model_dir, self.dictionary_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary file not found at: {path}")
        self.dictionary = corpora.Dictionary.load(path)

    def _load_corpus(self) -> None:
        path = os.path.join(self.processed_dir, self.corpus_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

    def _save_model(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, self.model_filename)
        self.lda_model.save(path)


if __name__ == "__main__":
    trainer : ModelTrainer = ModelTrainer(
        num_topics=6,
        passes=20,
        alpha="auto",
        eta="auto"
    )
    trainer.train()