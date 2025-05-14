from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import os
import pandas as pd
import streamlit as st
import fitz
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

OPENAI_API_KEY = "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"

def handle_csv(csv_path):
    df = pd.read_csv(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        agent = create_csv_agent(
            ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY),
            f,
            verbose=False,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )

    while True:
        question = input("Ask a question about your CSV or type 'exit':")
        if question.lower() in ["exit", "quit", "thank you"]:
            break
        response = agent.run(question)
        print(response)

def handle_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pdf_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pdf_text += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_text(pdf_text)
    documents = [Document(page_content=split) for split in splits]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)

    agent = ChatOpenAI(model_name="gpt-4o",temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = ConversationalRetrievalChain.from_llm(
        agent,
        vectorstore.as_retriever(),
        return_source_documents=True
    )

    chat_history = []
    while True:
        question = input("\nAsk a question about the PDF (or 'quit' to exit): ")
        if question.lower() in ["exit", "quit", "thank you"]:
            break

        result = chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(result["answer"])

def main():
    uploaded_file = input("Upload a CSV or PDF file: ")

    if uploaded_file.lower().endswith(".csv"):
        print("Processing CSV file...")
        handle_csv(uploaded_file)
    elif uploaded_file.lower().endswith(".pdf"):
        print("Processing PDF file...")
        handle_pdf(uploaded_file)
    else:
        print("Please upload a valid CSV or PDF file.")

if __name__ == "__main__":
    main()
