
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from loading_embedding_model import embedding_model
## INITIALIZE DATABASE 

FAISS_PATH = '/Users/nachanon/projects/rag_llm/faiss'

#this line of code below use to set dimension of embed vector that we want to store into vec db
index = faiss.IndexFlatL2(len(embedding_model().embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embedding_model(),
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)
#vector_store.add_documents(documents=docs_chunks_with_id, ids= [doc.metadata['id'] for doc in docs_chunks_with_id])

vector_store.save_local(FAISS_PATH)