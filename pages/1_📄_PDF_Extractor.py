import os

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.text_splitter import RecursiveCharacterTextSplitter #from langchain.llms import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings #from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from lib.constants import TEMPERATURE


st.set_page_config(page_title="PDF Extractor", page_icon="📄")

# Sidebar contents
with st.sidebar:

    st.markdown('''
    ### 📄 PDF Insights: No need to comb through pages. Extract crucial knowledge in seconds.
- upload a pdf file 
- ask questions about your pdf file
- get answers from your pdf file
- enjoy and support me with a star on [Github](https://www.github.com/Azzedde)

    ''')
    

load_dotenv()

def main():
    st.header("Chat with your pdf documents")

    # upload a pdf file
    pdf_docs = st.file_uploader("Upload your PDF", type='pdf', accept_multiple_files=True)
    
    #st.write(pdf)
    if pdf_docs is not None:
        text = ""
        for pdf in pdf_docs:
            #pdf reader
            pdf_reader = PdfReader(pdf)
            
            for page in pdf_reader.pages:
                text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = "pdf_store"
        st.write(f'{store_name}')
        # st.write(chunks)
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") #OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
 
        # embeddings = OpenAIEmbeddings()
        # vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = vector_store.similarity_search(query=query, k=5)
 
            llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=TEMPERATURE, convert_system_message_to_human=True) #OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            st.write(response)


if __name__ == '__main__':
    main()