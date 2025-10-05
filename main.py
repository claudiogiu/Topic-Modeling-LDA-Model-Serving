from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
import uvicorn
import re, logging, warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Topic Modeling API",
    description="A FastAPI-powered service for inferring topic distributions from scientific abstracts using a trained LDA model."
)

try:
    dictionary = corpora.Dictionary.load("models/dictionary.dict")
    lda_model = LdaModel.load("models/lda_model.gensim")
    logging.info("Model components loaded successfully: dictionary and trained LDA model.")
except Exception as e:
    logging.error(f"Failed to initialize model components: {e}")
    raise RuntimeError("Unable to initialize topic modeling components. Please verify model integrity and paths.")

topic_map = {
    0: "Statistics",
    1: "Physics",
    2: "Mathematics",
    3: "Computer Science",
    4: "Quantitative Finance",
    5: "Quantitative Biology"
}

class AbstractInput(BaseModel):
    abstract: str = Field(
        ...,
        example=(
            "Restricted Boltzmann Machines are described by the Gibbs measure of a bipartite spin glass, "
            "which in turn corresponds to the one of a generalised Hopfield network. This equivalence allows us to "
            "characterise the state of these systems in terms of retrieval capabilities, both at low and high load. "
            "We study the paramagnetic-spin glass and the spin glass-retrieval phase transitions, as the pattern "
            "(i.e. weight) distribution and spin (i.e. unit) priors vary smoothly from Gaussian real variables to "
            "Boolean discrete variables. Our analysis shows that the presence of a retrieval phase is robust and not "
            "peculiar to the standard Hopfield model with Boolean patterns. The retrieval region is larger when the "
            "pattern entries and retrieval units get more peaked and, conversely, when the hidden units acquire a broader "
            "prior and therefore have a stronger response to high fields. Moreover, at low load retrieval always exists "
            "below some critical temperature, for every pattern distribution ranging from the Boolean to the Gaussian case."
        )
    )

class TopicDistribution(BaseModel):
    topics: dict[str, float]

def preprocess_text(text: str) -> list[str]:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    return [
        lemmatizer.lemmatize(re.sub(r"[^A-Za-z]+", " ", token))
        for token in tokens if token not in stop_words
    ]


@app.post("/predict", response_model=TopicDistribution, status_code=status.HTTP_200_OK)
async def predict(input: AbstractInput):
    try:
        tokens = preprocess_text(input.abstract)
        if not tokens:
            logging.warning("Preprocessing yielded no valid tokens.")
            raise HTTPException(status_code=400, detail="Input preprocessing failed: no valid tokens extracted.")
        
        bow = dictionary.doc2bow(tokens)
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        mapped = {
            topic_map.get(tid, f"Topic_{tid}"): float(f"{prob:.4f}")
            for tid, prob in topic_dist
        }

        logging.info(f"Inference completed.")
        return {"topics": mapped}
    
    except Exception as e:
        logging.error(f"Topic inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error during topic inference. Please contact system administrator.")


@app.get("/health", tags=["Utility"])
async def service_health():
    return {"status": "OK"}


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)