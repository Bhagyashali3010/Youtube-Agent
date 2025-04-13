import streamlit as st
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import yt_dlp
import os
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import spacy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Ensure qa_chain is declared globally at the top of the script
qa_chain = None

# Helper functions
def get_youtube_video_metadata(video_url):
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build('youtube', 'v3', developerKey=api_key)

    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    video_id = video_id[0]

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()

    if not response['items']:
        raise ValueError("Video not found")

    item = response['items'][0]
    return {
        "video_id": video_id,
        "title": item["snippet"]["title"],
        "channel_title": item["snippet"]["channelTitle"],
        "published_at": item["snippet"]["publishedAt"],
        "description": item["snippet"]["description"],
        "view_count": item["statistics"].get("viewCount", 0),
        "like_count": item["statistics"].get("likeCount", 0),
        "tags": item["snippet"].get("tags", []),
    }

def download_youtube_audio(url, output_path="downloads"):
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return os.path.splitext(filename)[0] + ".mp3"

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

def summarize_transcript_with_gemini(transcript_text):
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes transcripts from audio or video."),
        HumanMessage(content=f"Summarize the following transcript in a concise paragraph:\n\n{transcript_text}")
    ]
    response = chat(messages)
    return response.content

def extract_named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def setup_qa_with_transcript(transcript_text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(transcript_text)]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    return RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)

def full_pipeline(youtube_url):
    metadata = get_youtube_video_metadata(youtube_url)
    file_path = download_youtube_audio(youtube_url)
    transcript = transcribe_audio(file_path)
    summary = summarize_transcript_with_gemini(transcript)
    entities = extract_named_entities(transcript)
    qa_chain = setup_qa_with_transcript(transcript)
    return {
        "metadata": metadata,
        "file": file_path,
        "transcript": transcript,
        "summary": summary,
        "entities": entities,
        "qa_chain": qa_chain
    }

# Streamlit UI
st.title("🎥 YouTube Video Analyzer")
youtube_url = st.text_input("Enter a YouTube video URL")

if youtube_url:
    with st.spinner("Processing video..."):
        try:
            # Declare qa_chain as global before assignment
            # global qa_chain
            result = full_pipeline(youtube_url)
            qa_chain = result['qa_chain']

            st.subheader("📄 Video Metadata")
            st.write(result["metadata"])

            st.subheader("🔊 Transcript")
            st.text_area("Transcript", result["transcript"], height=200)

            st.subheader("🧠 Summary")
            st.write(result["summary"])

            st.subheader("🔍 Named Entities")
            st.write(result["entities"])

            st.subheader("💬 Ask a question about the video")
        except Exception as e:
            st.error(f"Error: {e}")

# Handle question input separately
question = st.text_input("Type your question here")
if question:
    if qa_chain:
        try:
            answer = qa_chain.invoke({"query": question})
            st.markdown(f"**🤖 Answer:** {answer['result']}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please process a video first before asking a question.")
