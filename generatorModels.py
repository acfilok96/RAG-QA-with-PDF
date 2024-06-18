from groqApiKey import GroqApiKey
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def GeneratorModels():
    
    llm = ChatGroq(groq_api_key = str(GroqApiKey()),
                   model_name = "llama3-70b-8192",
                   temperature = 0.0)
    
    system_prompt = ChatPromptTemplate.from_template(
                                            """
                                            Answer the questions based on the provided context only.
                                            Please provide the most accurate response based on the question
                                            <context>
                                            {context}
                                            <context>
                                            Questions:{input}

                                            """
                                            )

    document_chain = create_stuff_documents_chain(llm, system_prompt)
    
    return document_chain