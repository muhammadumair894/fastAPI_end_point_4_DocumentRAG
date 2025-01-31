from fastapi import FastAPI, UploadFile, File, Body
import os
import nest_asyncio
import uvicorn
from pyngrok import ngrok
from pydantic import BaseModel
from typing import List
from PyPDF2 import PdfReader

# For demonstration, we assume these modules are installed and accessible.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub


app = FastAPI()

os.environ["GROQ_API_KEY"] =key
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = ChatGroq(model="llama3-8b-8192")  # Adjust accordingly.

#  Create global objects to store the vector store.
vectorstore = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
prompt = hub.pull("rlm/rag-prompt")


# The chain will need to retrieve from vectorstore, so we'll define a function:

class Question(BaseModel):
    question: str



@app.post("/upload-cv/")
async def upload_cv(cv: UploadFile = File(...)):
    """
    Upload a CV in PDF format, extract its text content, and store it in a vector database.

    Args:
        cv (UploadFile): The CV file uploaded by the user, expected to be in PDF format.

    Returns:
        dict: A success message with the number of document splits or an error message.

    Raises:
        HTTPException: If the file is not a valid PDF.
        Exception: For any other unforeseen errors during processing.
    """
    global vectorstore
    try:
        # Validate the file type to ensure it's a PDF
        if not cv.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported. Please upload a valid PDF file."
            )

        # Read the raw content of the uploaded file
        cv_content_raw = await cv.read()

        # Process the PDF file
        if cv.filename.endswith(".pdf"):
            # Save the uploaded PDF temporarily
            with open("temp.pdf", "wb") as temp_pdf:
                temp_pdf.write(cv_content_raw)

            #pdf_reader = PdfReader("temp.pdf")
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            splits = text_splitter.split_documents(docs)

            # Create or overwrite the vector store with the document chunks

            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

            # Return success message along with the number of chunks
            return {
                "message": "PDF uploaded successfully.",
                "num_docs": len(splits)
            }

    except HTTPException as e:
        # Raise an HTTPException if file validation fails
        raise e
    except Exception as e:
        # Catch any other errors and return the error message
        return {"error": str(e)}


@app.post("/ask_question")
async def ask_question(input_data: Question):
    global vectorstore
    if vectorstore is None:
        return {"error": "No PDF data found. Please upload a PDF first."}

    # Build a retriever from the updated vector store.
    retriever = vectorstore.as_retriever()

    # A helper to format docs into a single string.
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the chain.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Now invoke the chain with the user question.
    answer = rag_chain.invoke(input_data.question)

    return {"answer": answer}


if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
