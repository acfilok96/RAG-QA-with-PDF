from langchain_community.embeddings import HuggingFaceEmbeddings
from hf_token import Hf_Token

def EmbeddingModels():
    
    HF_TOKEN = str(Hf_Token())
    
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-l6-v2", # "mixedbread-ai/mxbai-embed-large-v1", # mixedbread-ai/mxbai-embed-large-v1
                                    encode_kwargs = {"normalize_embeddings" : False}, # "precision" : "binary", 
                                    model_kwargs = {"device" : "cpu"},
                                    )
    return embeddings