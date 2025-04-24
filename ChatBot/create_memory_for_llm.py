from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s)

DATA_PATH = "data/"

def load_pdf_files(data):   # this function loads the book
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls = PyPDFLoader)
    documents = loader.load();
    return documents; 

documents = load_pdf_files(data=DATA_PATH)
#print("length of pdf Book in pages: ",len(documents))  # prints 759 as there are 759 pages in the book uploaded.

# step 2 : Create Chunks 

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)  
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ",len(text_chunks))


# step 3: Create vector embeddings

def get_embedding_model():
    # the give model maps sentences & paragarphs to a 384-dimensional dense vector space and can be used for 
    # tasks like clustering or semantic search. In simple words it will convert the texts into
    # a vector  form
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
    return embedding_model

embedding_model= get_embedding_model()  

  
  # step 4: store embeddings in FAISS
  
DB_FAISS_PATH="vectorstore/db_faiss"  #database is stored in this folder
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
  