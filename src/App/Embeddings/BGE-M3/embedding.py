import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def get_vectorstore(splits):
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_name = "upskyy/bge-m3-korean"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_dir='weights',
    )

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vectorstore