import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def PdfLoader(pdf_files):
    
    text = f"""\n\n"""
      
    for j in pdf_files:
        loader = PyPDF2.PdfReader(j)
        for i in range(len(loader.pages)):
            page_obj = loader.pages[i]
            text_data = page_obj.extract_text()
            text = text + str(text_data) + str("\n\n")
        text = text + str("\n\n\n\n")
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
            
    documents = text_splitter.create_documents([text])
    
    return documents