# Topic Model Serving for LDA via FastAPI and Docker  

## Introduction  

This repository is designed for deploying a topic model via a REST API built with FastAPI. The implemented algorithm corresponds to Latent Dirichlet Allocation (LDA), originally introduced by BLEI D.M., NG A.Y., and JORDAN M.I. (2003) in their paper *"Latent Dirichlet Allocation"* (Journal of Machine Learning Research, Vol. 3, pp. 993–1022, DOI: [10.5555/944919.944937](https://dl.acm.org/doi/10.5555/944919.944937)).

LDA is a generative probabilistic model that identifies latent topics within a corpus by modeling documents as mixtures of thematic structures, each defined by a distribution over words. It enables unsupervised analysis of textual data through iterative estimation of hidden semantic patterns.

To ensure seamless deployment and reproducibility, the application is containerized with Docker, making it easy to run across different environments.

## Getting Started

To set up the repository properly, follow these steps:

**1.** **Configure the Environment File**  

- Initialize the environment configuration by copying the `.env.example` file template into the project root as `.env`:

  ```bash
  mv .env.example .env  
  ```

- Assign valid values to all required variables.

**2.** **Execute the Pipeline with Makefile**  

- The repository includes a **Makefile** to automate execution of all essential steps required for baseline functionality.

- Run the following command to execute the full workflow:

  ```bash
  make all
  ```

- This command sequentially performs the following operations:

  - Creates a Python virtual environment and installs all required dependencies.
  - Inizializes the `data/raw` and `data/processed` directories for source data ingestion and preprocessed data storage.
  - Retrieves the **Topic Modeling for Research Articles** dataset, available at [this link](https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles).
  - Preprocesses textual data, trains the LDA model, and computes evaluation metrics to assess its performance.
  - Builds the container image defined in the `Dockerfile` and launches the corresponding model serving service.

**3.** **Interact with the API** 
  
  Once the container is running, the API is accessible at:

  - **Swagger UI for interactive docs:** `localhost:8000/docs`  
  - **Health check endpoint:** `/health`  
  - **Prediction requests:** `/predict`

## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 
