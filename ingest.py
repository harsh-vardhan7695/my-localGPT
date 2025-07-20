import logging
import os

# Configure logging to suppress PostHog errors before any imports
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", 
    level=logging.INFO
)

# Set PostHog logger to CRITICAL level to suppress all errors
logging.getLogger("posthog").setLevel(logging.CRITICAL)

# Custom filter to suppress specific PostHog error messages
class PostHogFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == "posthog" and "Failed to send telemetry event" in record.getMessage())

# Apply filter to root logger
logging.getLogger().addFilter(PostHogFilter())

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Disable telemetry to prevent PostHog errors
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"

# Suppress PostHog telemetry warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="posthog")

import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def file_log(logentry):
    file1=open("file_ingest.log","a")
    file1.write(logentry+"\n")
    file1.close()
    print(logentry+"\n")


def load_single_document(file_path:str) -> Document:
    """Load a single document from a file path"""
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path+"loaded Successfully.")
            loader = loader_class(file_path)
        else:
            file_log(file_path+"not supported file type.")
            raise ValueError(f"Unsupported file type: {file_extension}")
        return loader.load()[0]
    except Exception as e:
        file_log(file_path+"Error loading file: "+str(e))
        return None
    
def load_document_batch(filepaths):
    logging.info(f"Loading documents in batch")
    #creating a thread pool executor
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        #collect the data
        if futures is None:
            file_log(name + "failed to submit")
            return None
        else:
            data_list=[future.result() for future in futures]
            return (data_list,filepaths)
        
def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    # Files to ignore during document processing
    ignore_files = {'.gitkeep', '.gitignore', '.DS_Store', 'Thumbs.db'}
    
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            # Skip files that should be ignored
            if file_name in ignore_files:
                print(f"Skipping: {file_name}")
                continue
                
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                # Filter out None values before extending docs
                valid_contents = [doc for doc in contents if doc is not None]
                docs.extend(valid_contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs

def split_documents(documents: list[Document]) -> tuple[list[Document],list[Document]]:
    """Split documents for correct text spltter"""
    text_docs,python_docs = [],[]
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata['source'])[1]
            if file_extension == '.py':
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs,python_docs

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)

def main(device_type:str):
    #load the documents and split them into chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")
    
    # Check if we have any valid documents to process
    if len(texts) == 0:
        logging.error("No valid documents found to create embeddings. Please check your source documents.")
        return

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within fun_localGPT.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )


if __name__ == "__main__":
    main()










