import time
import streamlit as st
from generatorModels import GeneratorModels
from embeddingModels import EmbeddingModels
from pdfLoader import PdfLoader
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title='RAG', layout="wide", initial_sidebar_state = 'auto')

st.markdown("""
<style>
.big-font-1 {
    font-size:30px !important;
    text-align: center; 
    color: yellow
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font-1">RAG: ChatPDF</p>', unsafe_allow_html = True)
st.info("""

        Introducing the RAG application for PDF documents. Upload PDF files and search \
        queries to find information within documents.

""")

with st.sidebar:
    st.header("**Upload Document**")
    with st.container(height = 200, border = False):
        pdf_files = st.file_uploader("Upload PDF file", type = ["pdf"], accept_multiple_files = True)
        
if "vectors" not in st.session_state:
    st.session_state.vectors = None
    
if pdf_files:
    if pdf_files:

        # with st.spinner(":green[Data Loading . . . ]"):
        final_documents = PdfLoader(pdf_files)
        st.sidebar.info("Step 1: Data Uploading Completed !")
        
        with st.spinner(":green[Analyzing . . . ]"):
            # time.sleep(0.1)
            embeddings = EmbeddingModels()
            st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
        st.sidebar.success("Step 2: Indexing Completed !")

    retriever = st.session_state.vectors.as_retriever()
    document_chain = GeneratorModels()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)


    st.sidebar.warning("Step 3: Continue Conversation !")
    with st.container(height = 400, border = False):
       user_prompt = st.text_input("Enter query: ", placeholder = "Enter query", help = "For example, Provide me the objective function of the GAN model.")
       if st.button("Get Response"):
           if user_prompt:
               try:
                   with st.spinner(":green[Wait . . . ]"):
                       response = retrieval_chain.invoke({"input" : user_prompt})
                       st.write(response["answer"])
               except Exception as e:
                   st.write("Ask once again!")
                
        
else:
    st.session_state.vectors = None
    st.warning("👈 Enter document and continue conversation !")
