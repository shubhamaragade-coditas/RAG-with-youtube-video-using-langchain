from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ai21 import AI21Embeddings
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

def extract_transcript_from_video(youtube_video_url: str) -> str:
    try:
        video_id: str = youtube_video_url.split("=")[1]

        transcript_text: list[dict] = YouTubeTranscriptApi.get_transcript(video_id)

        transcript: str = ""
        for text in transcript_text:
            transcript += " " + text["text"]

        return transcript

    except Exception as e:
        raise e
    
def text_to_chunks(transcript: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits text data into chunks of a specified size with optional overlap."""
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits: list[str] = text_splitter.split_text(transcript)
    return all_splits


def generate_embeddings(chunks: list[str]) -> FAISS:
    """Generates embeddings for a list of text chunks using an AI21Embeddings model."""
    embedding_model: AI21Embeddings = AI21Embeddings()
    vector_db: FAISS = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    return vector_db


def get_prompt() -> str:  # Specify return type as string
    """Retrieves a prompt from the langchain hub."""
    PROMPT = hub.pull("rlm/rag-prompt")
    return PROMPT


def get_answer_from_AI(
    vector_db: FAISS, question: str
) -> str:  # Specify input and output types
    """Gets an answer to a question using a RetrievalQA chain with the provided vector store and question."""

    PROMPT: str = get_prompt()
    llm: ChatGroq = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    qa_chain: RetrievalQA = RetrievalQA.from_chain_type(
        llm, retriever=vector_db.as_retriever(), chain_type_kwargs={"prompt": PROMPT}
    )

    result: dict[str, str] = qa_chain.invoke({"query": question})
    return result["result"]


youtube_video_url: str = input("Enter the url: ")

transcript: str = extract_transcript_from_video(youtube_video_url=youtube_video_url)

transcript_chunks: list[str] = text_to_chunks(transcript=transcript, chunk_size=100, chunk_overlap=20)

transcript_embddings: FAISS = generate_embeddings(chunks=transcript_chunks)

continue_asking: bool = True
while(continue_asking):
    question: str = input("Enter your question: ")
    answer: str = get_answer_from_AI(vector_db=transcript_embddings,question=question)
    print(f"User: {question}\nAI: {answer}")

    continue_asking = int(input("Do you want to continue:\n 0. No\n 1. Yes"))

