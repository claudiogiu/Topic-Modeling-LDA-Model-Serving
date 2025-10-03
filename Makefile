.PHONY: all setup dirs venv nltk run download preprocess train evaluate docker-build docker-run serve

# Python interpreter inside virtual environment
PYTHON=.venv/bin/python

# Master target: full pipeline from setup to API deployment
all: setup run serve
	@echo "[ALL] End-to-end execution completed."

# Step 1: Environment setup
setup: dirs venv nltk
	@echo "[SETUP] Environment initialization completed. Dependency management and resource provisioning finalized."

# Step 2: Directory creation
dirs:
	mkdir -p data/raw data/processed
	@echo "[DIRECTORY STRUCTURE] Data directories 'data/raw' and 'data/processed' successfully instantiated."

# Step 3: Virtual environment and dependency installation
venv:
	python3.12 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install \
		pandas==2.3.2 \
		python-dotenv==1.1.1 \
		gensim==4.3.3 \
		nltk==3.9.1 \
		kaggle==1.7.4.5
	@echo "[VENV] Virtual environment '.venv' created and essential packages installed."

# Step 4: NLTK resource download
nltk:
	$(PYTHON) -m nltk.downloader stopwords wordnet
	@echo "[NLTK] Linguistic resources 'stopwords' and 'wordnet' successfully retrieved."

# Step 5: Script execution sequence
run: download preprocess train evaluate

download:
	$(PYTHON) src/load_data.py
	@echo "[DATA INGESTION] Dataset retrieved from Kaggle and stored in 'data/raw'."

preprocess:
	$(PYTHON) src/preprocess.py
	@echo "[PREPROCESSING] Text normalization, tokenization, and corpus construction completed. Dictionary and corpus saved."

train:
	$(PYTHON) src/train_model.py
	@echo "[TRAINING] LDA model trained and serialized to disk."

evaluate:
	$(PYTHON) src/evaluate_model.py
	@echo "[EVALUATION] Performance metrics computed."

# Step 6: Docker container build and deployment
docker-build:
	docker build -t topic-api .
	@echo "[DOCKER BUILD] Container image 'topic-api' successfully built."

docker-run:
	docker run -p 8000:8000 topic-api
	@echo "[DOCKER RUN] Container launched. API available at http://localhost:8000."

serve: docker-build docker-run
	@echo "[SERVING] API deployment finalized. Inference interface is now active."