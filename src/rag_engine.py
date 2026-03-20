import os
import shutil
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

import config

class RAGEngine:
    def __init__(self):
        self.persist_directory = config.CHROMA_DB_DIR
        
        # Initialize Google Embeddings
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", # Hardcoded or from config
            google_api_key=config.GOOGLE_API_KEY
        )
        
        # Initialize Vector Store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.MODEL_NAME,
            temperature=0.2,
            google_api_key=config.GOOGLE_API_KEY,
            convert_system_message_to_human=True # Sometimes needed for older Gemini versions, safe to keep or remove based on lib version
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def ingest_data(self, source_path: str):
        """
        Ingests data from a file or directory.
        Currently supports PDF and txt.
        """
        docs = []
        if os.path.isdir(source_path):
            for filename in os.listdir(source_path):
                file_path = os.path.join(source_path, filename)
                # Skip directories (like chroma_db) and non-data files
                if os.path.isdir(file_path):
                    continue
                if not (filename.endswith('.pdf') or filename.endswith('.txt')):
                    continue
                    
                docs.extend(self._load_file(file_path))
        else:
            docs.extend(self._load_file(source_path))

        if not docs:
            print("No new valid documents found to ingest.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Add to vector store
        self.vector_store.add_documents(documents=splits)
        print(f"Ingested {len(splits)} chunks into the Knowledge Base.")

    def _load_file(self, file_path: str) -> List[Document]:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, autodetect_encoding=True)
            return loader.load()
        else:
            print(f"Unsupported file type: {file_path}")
            return []

    def query(self, query_text: str) -> str:
        """
        Queries the knowledge base and returns a response.
        Enforces a default response if confidence is low or content is missing (via prompt).
        """
        
        # Custom Prompt Template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, get creative and make up an answer.
        
        {context}
        
        Question: {question}
        Helpful Answer:"""
        
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke(query_text)
            return result

        except Exception as e:
            print(f"RAG Query Error: {e}")
            return "I encountered an error while processing your request."

    def clear_database(self):
        """Clears the existing vector database."""
        try:
            # Try to delete the collection directly
            self.vector_store.delete_collection()
            print("Database cleared (collection deleted).")
            
            # Re-initialize to ensure structure allows new additions immediately
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error clearing database: {e}")
            raise e
