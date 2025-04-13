# YouTube Video Analyzer

A powerful Python application that analyzes YouTube videos, transcribes their content, and provides interactive Q&A capabilities using AI.

## Features

- YouTube video metadata extraction
- Audio download and transcription
- AI-powered video summary generation
- Named entity recognition
- Interactive Q&A system based on video content
- User-friendly Streamlit interface

## Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Bhagyashali3010/Youtube-Agent.git
cd Youtube-Agent
```

2. Create and activate a virtual environment:
```bash
python -m venv yt_env
# On Windows
yt_env\Scripts\activate
# On Unix or MacOS
source yt_env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following:
```
YOUTUBE_API_KEY=your_youtube_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Required API Keys

- **YouTube API Key**: Get it from [Google Cloud Console](https://console.cloud.google.com/)
- **Google API Key**: Required for Gemini Pro access

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter a YouTube URL in the input field and wait for the processing to complete

4. Explore the following sections:
   - Video Metadata
   - Video Transcript
   - AI-Generated Summary
   - Named Entities
   - Interactive Q&A

## Components

- `streamlit`: Web interface
- `google-api-python-client`: YouTube data API integration
- `yt-dlp`: YouTube video download
- `whisper`: Audio transcription
- `langchain`: AI chain operations and Q&A system
- `spacy`: Named entity recognition
- `FAISS`: Vector storage for efficient text search
- `google-generativeai`: Gemini Pro integration

## Project Structure

```
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
└── downloads/          # Downloaded audio files
```

## Functions Overview

- `get_youtube_video_metadata()`: Fetches video information using YouTube API
- `download_youtube_audio()`: Downloads and converts video to MP3
- `transcribe_audio()`: Generates transcript using Whisper
- `summarize_transcript_with_gemini()`: Creates AI summary
- `extract_named_entities()`: Identifies key entities in the transcript
- `setup_qa_with_transcript()`: Prepares Q&A system
- `full_pipeline()`: Orchestrates the entire analysis process

## Error Handling

The application includes comprehensive error handling for:
- Invalid YouTube URLs
- API failures
- File processing issues
- Q&A system errors

## Performance Considerations

- Audio files are stored locally in the downloads/ directory
- Transcription may take several minutes depending on video length
- Q&A system requires initial processing before questions can be asked


