from dataclasses import dataclass, field
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env before importing KaggleApi.
# Required because KaggleApi reads credentials from os.environ at import time.
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi
import warnings

warnings.filterwarnings("ignore")

@dataclass
class KaggleDataDownloader:
    """
    Interface for downloading Kaggle datasets, enforcing the presence of environment variables for slug and credentials.

    Attributes:
        slug (str): Dataset identifier, sourced from environment variables at runtime.

    Methods:
        __post_init__() -> None:
            Loads the dataset slug from environment and validates its presence.

        download() -> None:
            Enforces credential presence and downloads the dataset into the raw data directory.

        _get_raw_path() -> str:
            Resolves the absolute path to the 'data/raw' directory relative to the project root.
    """

    slug: str = field(init=False)

    def __post_init__(self):
        self.slug = os.getenv("KAGGLE_DATASET_SLUG", "")
        if not self.slug:
            raise EnvironmentError("Missing required .env variable: KAGGLE_DATASET_SLUG")

    def download(self) -> None:
        if not os.environ["KAGGLE_USERNAME"] or not os.environ["KAGGLE_KEY"]:
            raise EnvironmentError("Missing required credentials: KAGGLE_USERNAME or KAGGLE_KEY")

        try:
            api = KaggleApi()
            api.authenticate()

            raw_path = self._get_raw_path()
            os.makedirs(raw_path, exist_ok=True)

            print(f"Downloading dataset '{self.slug}' into '{raw_path}'...")
            api.dataset_download_files(self.slug, path=raw_path, unzip=True)

            print("Download completed.")
        except Exception as e:
            print(f"Error during Kaggle download: {e}")

    def _get_raw_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(project_root, "data", "raw")


@dataclass
class DataLoader:
    """
    Interface for loading CSV datasets from the local filesystem.

    Attributes:
        data_type (str): Subdirectory under 'data/' indicating dataset stage ('raw' or 'processed').
        dataset_name (str): Filename of the dataset to load.

    Methods:
        load() -> pd.DataFrame | None:
            Loads the specified CSV file into a DataFrame, returning None if the file is missing or unreadable.

        _resolve_path() -> str:
            Constructs the absolute path to the target dataset file based on the project structure.
    """

    data_type: str
    dataset_name: str

    def load(self) -> pd.DataFrame | None:
        file_path: str = self._resolve_path()

        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)
            print(f"Dataset '{self.dataset_name}' successfully loaded from 'data/{self.data_type}'. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Failed to load '{self.dataset_name}': {e}")
            return None

    def _resolve_path(self) -> str:
        root = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(root, "data", self.data_type, self.dataset_name)


if __name__ == "__main__":
    downloader: KaggleDataDownloader = KaggleDataDownloader()
    downloader.download()