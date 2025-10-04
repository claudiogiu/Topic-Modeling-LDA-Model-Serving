# Topic Model Serving for LDA via FastAPI and Docker  

## Introduction  

This repository is designed for serving a topic modeling solution via a REST API built with FastAPI, leveraging Latent Dirichlet Allocation (LDA), originally introduced by BLEI D.M., NG A.Y., and JORDAN M.I. (2003) in their paper *"Latent Dirichlet Allocation"* (Journal of Machine Learning Research, Vol. 3, pp. 993–1022, DOI: [10.5555/944919.944937](https://dl.acm.org/doi/10.5555/944919.944937))
.

LDA is a generative probabilistic model that identifies latent topics within a corpus by modeling documents as mixtures of thematic structures, each defined by a distribution over words. It enables unsupervised analysis of textual data through iterative estimation of hidden semantic patterns.

To ensure seamless deployment and reproducibility, the application is containerized with Docker, making it easy to run across different environments.

## Getting Started

To properly set up the repository, run the following commands from the project root:

```bash
mv .env.example .env  
# Assign valid values to all required variables in .env  
make all
```

This single step performs the entire pipeline setup and execution:

- Downloads the dataset **Topic Modeling for Research Articles**, available at [this link](https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles).

- Creates the `data/` folder in the project root with the following subdirectories:
  - `raw/`: This will store the unprocessed dataset  
  - `processed/`: stores the training and test splits

- Executes all pipeline modules in `src/`:
  - `"load_data.py"`: Ingests and validates the dataset.  
  - `"preprocess.py"`: Applies text preprocessing and stores vectorizers in `models/`  
  - `"train_model.py"`: Trains the LDA model and serializes it to `models/`  
  - `"evaluate_model.py"`: Computes validation metrics

- Prepares the trained model for serving via a REST API implemented in `main.py` using FastAPI.

  Once the container is running, the API is accessible at:

  - **Swagger UI for interactive documentation:** `localhost:8000/docs`  
  - **Health check endpoint:** `/health`  
  - **Prediction requests** `/predict`

## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 
