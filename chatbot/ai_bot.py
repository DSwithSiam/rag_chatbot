import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

class RAGChatbot:
    def __init__(self, file_path):
        self.file_path = file_path
        self.chain = self._initialize()

    def _initialize(self):
        # Load PDF
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        # Embeddings + FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # LLM
        llm = ChatOpenAI(temperature=0)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        prompt = """
You are a strict AI assistant.

Rules:
- Only answer from the provided context
- Do NOT guess
- If answer not found, return EXACTLY:
"This information is not present in the provided document."

Context:
{context}

Question:
{question}
"""

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )

        chain.combine_docs_chain.llm_chain.prompt.template = prompt

        return chain

    def ask(self, question):
        result = self.chain.invoke({"question": question})
        return {
            "answer": result["answer"],
            "sources": [doc.metadata.get("source") for doc in result["source_documents"]]
        }


# Singleton instance (important)
chatbot_instance = RAGChatbot("data/document.pdf")