import os
import streamlit as st
from document_loader import load_document, load_pdf_document, load_json_document
from retriever import retrieve_documents
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

def main():
    st.title("Q&A Bot")

    # Get the file type from the user
    file_type = st.selectbox("Select the file type", ["pdf", "json", "txt"])

    # Upload the file
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "json", "txt"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_path = f"temp_file.{file_type}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the document based on the specified file type
        if file_type == 'pdf':
            document = load_pdf_document(file_path)
        elif file_type == 'json':
            document = load_json_document(file_path)
        elif file_type == 'txt':
            document = load_document(file_path)

        documents = [document]

        # Initialize the ChatGroq instance
        llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.1-70b-versatile"
        )

        # Create a prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are an expert assistant. Answer the question based on the provided context.

            Context: {context}
            Question: {question}

            Answer:
            """
        )

        # Initialize a string output parser
        output_parser = StrOutputParser()

        # User can ask questions
        question = st.text_input("Ask a question about the document:")
        if question:
            relevant_documents = retrieve_documents(question, documents)
            context = " ".join(relevant_documents) if relevant_documents else "No relevant context found."
            prompt = prompt_template.format(question=question, context=context)

            # Get an answer from the LLM
            try:
                response = llm.invoke([{"role": "user", "content": prompt}])
                parsed_response = output_parser.parse(response.content)
                st.write("Answer:", parsed_response)
            except Exception as e:
                st.error(f"Error getting answer from ChatGroq: {e}")

if __name__ == "__main__":
    main()
