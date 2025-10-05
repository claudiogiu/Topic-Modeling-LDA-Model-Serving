from dataclasses import dataclass, field
import os
import json
import pandas as pd
import ast
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import warnings

warnings.filterwarnings("ignore")

@dataclass
class ModelEvaluator:
    """
    Interface for evaluating a trained LDA model using fixed test corpus and training texts.

    Attributes:
        dictionary_filename (str): Name of the dictionary file located in the model artifact directory.
        model_filename (str): Name of the trained LDA model file located in the model artifact directory.
        model_dir (str): Directory where model artifacts are stored.
        processed_dir (str): Directory containing preprocessed corpus and data.
        corpus_filename (str): Name of the test corpus file in JSON format.
        texts_filename (str): Name of the training CSV file containing tokenized texts.

        dictionary (corpora.Dictionary): Loaded dictionary object used for inference.
        corpus (list): List of test documents in BoW format.
        lda_model (LdaModel): Loaded LDA model instance used for evaluation.

    Methods:
        evaluate() -> dict:
            Loads dictionary, corpus, model and tokenized documents, computes coherence and per-word log perplexity, and returns evaluation metrics.

        _load_dictionary() -> None:
            Loads a previously saved dictionary from disk.

        _load_corpus() -> None:
            Loads the fixed test corpus from disk in JSON format.

        _load_model() -> None:
            Loads a trained LDA model from disk.

        _load_texts() -> list[list[str]]:
            Loads the training CSV and extracts the 'texts' column as a list of tokenized documents.
    """

    dictionary_filename: str = "dictionary.dict"
    model_filename: str = "lda_model.gensim"
    model_dir: str = "models"
    processed_dir: str = "data/processed"
    corpus_filename: str = "test_corpus.json"
    texts_filename: str = "train.csv"

    dictionary: corpora.Dictionary = field(init=False)
    corpus: list = field(init=False)
    lda_model: LdaModel = field(init=False)

    def evaluate(self) -> dict:
        self._load_dictionary()
        self._load_corpus()
        self._load_model()
        texts = self._load_texts()

        coherence_model = CoherenceModel(model=self.lda_model, texts=texts, dictionary=self.dictionary, coherence="c_v")
        coherence = coherence_model.get_coherence()
        log_perplexity = self.lda_model.log_perplexity(self.corpus)

        print(f"C_V Coherence Score on training texts: {coherence:.4f}")
        print(f"Log Perplexity on test corpus: {log_perplexity:.4f}")

        return {
            "coherence": coherence,
            "perplexity": log_perplexity,
        }

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

    def _load_model(self) -> None:
        path = os.path.join(self.model_dir, self.model_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LDA model file not found at: {path}")
        self.lda_model = LdaModel.load(path)

    def _load_texts(self) -> list[list[str]]:
        path = os.path.join(self.processed_dir, self.texts_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessed training CSV not found at: {path}")
        df = pd.read_csv(path)
        if "texts" not in df.columns:
            raise ValueError("Column 'texts' not found in training CSV.")
        return df["texts"].apply(ast.literal_eval).tolist()


if __name__ == "__main__":
    evaluator : ModelEvaluator = ModelEvaluator()
    metrics = evaluator.evaluate()
