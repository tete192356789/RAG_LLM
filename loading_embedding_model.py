from langchain.embeddings import OllamaEmbeddings
def embedding_model(model = "all-minilm"):
    embed_model = OllamaEmbeddings(model=model)
    return embed_model
    
    #embeddings.embed_documents(docs)