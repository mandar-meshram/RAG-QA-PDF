import os 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain_community.llms import HuggingFacePipeline
import tempfile
from .mongodb import MongoDBConnection
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        )
        # for OpenAI,
        # self.embeddings = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))

        self.vector_store = None
        self.qa_chain = None
        self.llm = None

    def _setup_free_llm(self):
        if self.llm is None:
            model_name = "microsoft/DialoGPT-medium"

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # check if GPU is available
            device = 0 if torch.cuda.is_available() else -1

            # create a pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length = 1000,
                do_sample = True,
                temperature = 0.7,
                pad_token_id = tokenizer.eos_token_id, 
                truncation = True
            )

            #wrap in langchain
            self.llm = HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs = {
                        "temperature": 0.7,
                        "max_length": 1000,
                        "do_sample": True
                    }
            )
            print(f"Successfully loaded free LLM: {model_name}")



    def process_pdf(self, pdf_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in pdf_file.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path) # loading the pdf
            documents = loader.load()

            # Splitting the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )
            chunks = text_splitter.split_documents(documents)

            # storing chunks in mongodb
            collection = MongoDBConnection.get_collection('document_chunks')

            # clear any existing chunks
            collection.delete_many({'document_name' : pdf_file.name})

            # insert new chunks
            for i , chunk in enumerate(chunks):
                chunk_data = {
                    'document_name' : pdf_file.name,
                    'chunk_index' : i,
                    'content' : chunk.page_content,
                    'metadata' : chunk.metadata
                }
                collection.insert_one(chunk_data)

            # create  vector store
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

            # create QA chain with paid openai key
            llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

            self.qa_chain = RetrievalQA.from_chain_type(
                llm = llm,
                chain_type="stuff",
                retriever = self.vector_store.as_retriever(search_kwargs = {"k" : 3})
            )

            return len(chunks)
            # self._setup_free_llm()    # free huggingface model

            # # create QA chain
            # self.qa_chain = RetrievalQA.from_chain_type(
            #     llm = self.llm,
            #     chain_type="stuff",
            #     retriever = self.vector_store.as_retriever(search_kwargs = {"k" : 3})
            # )
            # return len(chunks)
        
        finally:
            os.unlink(tmp_file_path)


    def query_document(self, question):
        if not self.qa_chain:
            return "Please upload a PDF document first."
        
        try:
            result = self.qa_chain.run(question)
            return result
        except Exception as e:
            return f"Error processing query : {str(e)}"
        

    def get_similar_chunks(self, query, k = 3):
        if not self.vector_store:
            return []
        
        similar_docs = self.vector_store.similarity_search(query, k = k)
        return [doc.page_content for doc in similar_docs]



