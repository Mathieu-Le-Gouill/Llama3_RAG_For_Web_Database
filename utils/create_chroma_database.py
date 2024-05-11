from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import bs4

embeddings = OllamaEmbeddings(model="avr/sfr-embedding-mistral")

# Open the file and read the URLs
with open('URLlinks.txt', 'r') as file:
    webURLs = [line.strip() for line in file.readlines()]

# Construct retriever
loader_web = WebBaseLoader(
    web_paths=tuple(webURLs),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("entry-header", "entry-content")
        )
    ),
)

loader_pdf = PyPDFDirectoryLoader("pdf_data/")

docsPDF = loader_pdf.load()
docsWEB = loader_web.load()

merged_loader = MergedDataLoader(loaders=[loader_web, loader_pdf])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = merged_loader.load_and_split(text_splitter)

# Store the database on disk (only needed once) then comment it
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="DataBase/chroma_db")