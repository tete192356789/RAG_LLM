from loading_embedding_model import embedding_model
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import os
import argparse

def data_loading(DATA_PATH):
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def documents_splitter(docs:list , chunk_size:int=500 ,chunk_overlap:int=100):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    )
    return splitter.split_documents(docs)

def generate_chunk_id(docs_chunks):
    current_page = 0
    doc_idx = 0
    
    for doc in docs_chunks:
        page = doc.metadata['page']
        source = doc.metadata['source']
        filename = os.path.basename(source)

        #page didnt change
        if page == current_page:
            
            #chunks_id.append(f'{page}-{doc_idx}')
            doc.metadata['id'] = f'{filename}:{page}-{doc_idx}'
            doc_idx += 1   
        #page changed    
        else:
            #source changed
            if page == 0 :
                current_page = 0
                doc_idx = 0
                doc.metadata['id'] = f'{filename}:{page}-{doc_idx}'
                doc_idx += 1
            #source didnt change but page 
            else:
                current_page += 1
                doc_idx = 0
                #chunks_id.append(f'{page}-{doc_idx}')
                doc.metadata['id'] = f'{filename}:{page}-{doc_idx}'
                doc_idx += 1
    return docs_chunks

def update_docs_db(vector_database, docs_chunks_with_id):
    not_found_docs = []
    for doc in docs_chunks_with_id:
        if type(vector_database.docstore.search(doc.metadata['id'])) == str:
            not_found_docs.append(doc)


    ######check point for today : update db : next-> more functional coding
    if len(not_found_docs) != 0 :
        vector_database.add_documents(documents=not_found_docs, ids= [doc.metadata['id'] for doc in not_found_docs])
        print(f'Update {len(not_found_docs)} documents.')
    else:
        print('no docs to update!')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RAG LLM.')
    parser.add_argument("--query", type=str, default="", help="Query prompting")
    args = parser.parse_args()

    query = args.query
    #query = "how much money each player got in starting"


    if query != "":
        DATA_PATH = "/Users/nachanon/projects/rag_llm/data"
        docs = data_loading(DATA_PATH)
        docs_chunks = documents_splitter(docs)
        embed_model = embedding_model()
        docs_chunks_with_id = generate_chunk_id(docs_chunks)
        
        FAISS_PATH = '/Users/nachanon/projects/rag_llm/faiss'
        loaded_vector_db = FAISS.load_local(FAISS_PATH,embedding_model(),allow_dangerous_deserialization=True)
        update_docs_db(loaded_vector_db,docs_chunks_with_id)

        prompt_template = """
        This is query from user: {question}
        And use this context to answer query : {context}

        """
        results = loaded_vector_db.similarity_search_with_relevance_scores(query)

        context_text = "\n\n-------------\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt_template.format(context=context_text, question=query)
        print(prompt)
        model = OllamaLLM(model="llama3")
        res = model.invoke(prompt)
        print('\nCompletion from LLM :\n\n')
        print(res)
    else:
        print("Query was Empty .")