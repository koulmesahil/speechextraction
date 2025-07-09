import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yt_dlp
import openai
from pydub import AudioSegment
import io
import json
import time
import re
from textblob import TextBlob
import spacy
from collections import Counter
from pyannote.audio import Pipeline
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="YouTube Speaker Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .speaker-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .timeline-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

# API Keys (you'll need to set these in Streamlit secrets)
# openai.api_key = st.secrets.get('openai_api_key', '')
# hf_api_key = st.secrets.get('hf_api_key', '')

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@st.cache_data
def download_youtube_audio(youtube_url):
    """Download audio from YouTube using yt-dlp"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': './temp_audio.%(ext)s',
            'extractaudio': True,
            'audioformat': 'mp3',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
        # Load audio with pydub
        audio = AudioSegment.from_file('./temp_audio.mp3')
        return audio, title, duration
    
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None, None, None

def perform_diarization(audio):
    """Perform speaker diarization using pyannote"""
    # This is a placeholder for the actual diarization
    # In reality, you'd use the pyannote pipeline here
    
    # Simulated diarization results for demo
    duration = len(audio) / 1000  # Convert to seconds
    
    # Create mock speaker segments
    segments = []
    current_time = 0
    speakers = ['Speaker_A', 'Speaker_B', 'Speaker_C']
    
    while current_time < duration:
        speaker = np.random.choice(speakers)
        segment_duration = np.random.uniform(3, 15)  # 3-15 second segments
        end_time = min(current_time + segment_duration, duration)
        
        segments.append({
            'speaker': speaker,
            'start': current_time,
            'end': end_time,
            'duration': end_time - current_time
        })
        
        current_time = end_time + np.random.uniform(0, 2)  # Small gaps between segments
    
    return segments

def transcribe_segments(audio, segments):
    """Transcribe audio segments using OpenAI Whisper"""
    # This is a placeholder for actual transcription
    # In reality, you'd use OpenAI's Whisper API here
    
    sample_texts = [
        "Welcome everyone to today's discussion about artificial intelligence.",
        "I think we should consider the implications of AI in healthcare.",
        "That's a great point. Let me share some statistics about AI adoption.",
        "The market research shows significant growth in this sector.",
        "I agree with the previous speaker about the importance of ethics.",
        "Let's move on to the next topic on our agenda.",
        "Has anyone had experience implementing AI solutions?",
        "The technical challenges are quite significant in this space.",
        "We need to consider the regulatory environment as well.",
        "Thank you for that insight. Any other questions?"
    ]
    
    transcribed_segments = []
    for i, segment in enumerate(segments):
        # Simulate transcription with random sample texts
        text = sample_texts[i % len(sample_texts)]
        
        transcribed_segments.append({
            'speaker': segment['speaker'],
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['duration'],
            'text': text
        })
    
    return transcribed_segments

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

def extract_entities(text):
    """Extract named entities (placeholder - would use spaCy in real implementation)"""
    # This is a simplified version - in reality you'd use spaCy
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Mock entities for demo
    sample_entities = [
        ("artificial intelligence", "TECHNOLOGY"),
        ("healthcare", "INDUSTRY"),
        ("market research", "CONCEPT"),
        ("AI solutions", "TECHNOLOGY")
    ]
    
    return sample_entities

def create_interactive_timeline(segments_data):
    """Create an interactive timeline visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Speaker Timeline', 'Sentiment Flow'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Color mapping for speakers
    speaker_colors = {
        'Speaker_A': '#1f77b4',
        'Speaker_B': '#ff7f0e', 
        'Speaker_C': '#2ca02c'
    }
    
    # Create speaker timeline
    for segment in segments_data:
        fig.add_trace(
            go.Scatter(
                x=[segment['start'], segment['end']],
                y=[segment['speaker'], segment['speaker']],
                mode='lines+markers',
                line=dict(width=20, color=speaker_colors.get(segment['speaker'], '#333')),
                name=segment['speaker'],
                showlegend=False,
                hovertemplate=f"<b>{segment['speaker']}</b><br>" +
                            f"Time: {segment['start']:.1f}s - {segment['end']:.1f}s<br>" +
                            f"Text: {segment['text']}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Create sentiment flow
    times = [segment['start'] for segment in segments_data]
    sentiments = [analyze_sentiment(segment['text'])[1] for segment in segments_data]
    
    fig.add_trace(
        go.Scatter(
            x=times,
            y=sentiments,
            mode='lines+markers',
            line=dict(width=3, color='purple'),
            marker=dict(size=8),
            name='Sentiment',
            showlegend=False,
            hovertemplate="Time: %{x:.1f}s<br>Sentiment: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Interactive Speaker Timeline & Sentiment Analysis",
        height=600,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Speakers", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1, range=[-1, 1])
    
    return fig

def create_speaker_analytics_dashboard(segments_data):
    """Create speaker analytics dashboard"""
    # Calculate speaking statistics
    speaker_stats = {}
    total_duration = sum(segment['duration'] for segment in segments_data)
    
    for segment in segments_data:
        speaker = segment['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'total_time': 0,
                'segments': 0,
                'words': 0,
                'sentiments': []
            }
        
        speaker_stats[speaker]['total_time'] += segment['duration']
        speaker_stats[speaker]['segments'] += 1
        speaker_stats[speaker]['words'] += len(segment['text'].split())
        
        sentiment_score = analyze_sentiment(segment['text'])[1]
        speaker_stats[speaker]['sentiments'].append(sentiment_score)
    
    # Calculate percentages and averages
    for speaker in speaker_stats:
        speaker_stats[speaker]['percentage'] = (speaker_stats[speaker]['total_time'] / total_duration) * 100
        speaker_stats[speaker]['avg_sentiment'] = np.mean(speaker_stats[speaker]['sentiments'])
        speaker_stats[speaker]['words_per_minute'] = speaker_stats[speaker]['words'] / (speaker_stats[speaker]['total_time'] / 60)
    
    return speaker_stats

def display_speaker_dashboard(speaker_stats):
    """Display the speaker analytics dashboard"""
    st.subheader("📊 Speaker Analytics Dashboard")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Speakers", len(speaker_stats))
    
    with col2:
        total_words = sum(stats['words'] for stats in speaker_stats.values())
        st.metric("Total Words", total_words)
    
    with col3:
        avg_sentiment = np.mean([stats['avg_sentiment'] for stats in speaker_stats.values()])
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    
    # Speaker breakdown
    st.subheader("🎤 Individual Speaker Analysis")
    
    for speaker, stats in speaker_stats.items():
        with st.expander(f"{speaker} - {stats['percentage']:.1f}% of conversation"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Speaking Time", f"{stats['total_time']:.1f}s")
            with col2:
                st.metric("Segments", stats['segments'])
            with col3:
                st.metric("Words/Min", f"{stats['words_per_minute']:.1f}")
            with col4:
                sentiment_emoji = "😊" if stats['avg_sentiment'] > 0.1 else "😐" if stats['avg_sentiment'] > -0.1 else "😔"
                st.metric("Avg Sentiment", f"{stats['avg_sentiment']:.2f} {sentiment_emoji}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Speaking time pie chart
        speakers = list(speaker_stats.keys())
        percentages = [speaker_stats[speaker]['percentage'] for speaker in speakers]
        
        fig_pie = px.pie(
            values=percentages,
            names=speakers,
            title="Speaking Time Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment comparison
        speakers = list(speaker_stats.keys())
        sentiments = [speaker_stats[speaker]['avg_sentiment'] for speaker in speakers]
        
        fig_bar = px.bar(
            x=speakers,
            y=sentiments,
            title="Average Sentiment by Speaker",
            color=sentiments,
            color_continuous_scale="RdYlBu"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">🎙️ YouTube Speaker Analytics</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Analyze conversations from YouTube videos
    - **Interactive Timeline**: Visualize speaker segments and sentiment flow
    - **Speaker Analytics**: Get detailed statistics about each speaker
    - **Sentiment Analysis**: Track emotional tone throughout the conversation
    - **Entity Recognition**: Extract key topics and concepts
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # YouTube URL input
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        # Processing options
        st.subheader("Processing Options")
        max_duration = st.slider("Max Duration (minutes)", 1, 10, 5)
        include_sentiment = st.checkbox("Include Sentiment Analysis", True)
        include_entities = st.checkbox("Include Entity Recognition", True)
        
        # Process button
        if st.button("🚀 Process Video", type="primary"):
            if youtube_url:
                process_video(youtube_url, max_duration, include_sentiment, include_entities)
            else:
                st.error("Please enter a YouTube URL")
    
    # Main content area
    if st.session_state.processed_data:
        display_results()
    else:
        st.info("👆 Enter a YouTube URL in the sidebar to get started!")

def process_video(youtube_url, max_duration, include_sentiment, include_entities):
    """Process the YouTube video"""
    with st.spinner("Processing video..."):
        # Download audio
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Downloading audio...")
        progress_bar.progress(20)
        
        audio, title, duration = download_youtube_audio(youtube_url)
        if not audio:
            return
        
        # Limit duration
        if duration > max_duration * 60:
            audio = audio[:max_duration * 60 * 1000]
            duration = max_duration * 60
        
        status_text.text("Performing speaker diarization...")
        progress_bar.progress(40)
        
        # Perform diarization
        segments = perform_diarization(audio)
        
        status_text.text("Transcribing segments...")
        progress_bar.progress(60)
        
        # Transcribe segments
        transcribed_segments = transcribe_segments(audio, segments)
        
        status_text.text("Analyzing content...")
        progress_bar.progress(80)
        
        # Add sentiment analysis
        if include_sentiment:
            for segment in transcribed_segments:
                sentiment, score = analyze_sentiment(segment['text'])
                segment['sentiment'] = sentiment
                segment['sentiment_score'] = score
        
        # Add entity recognition
        if include_entities:
            for segment in transcribed_segments:
                entities = extract_entities(segment['text'])
                segment['entities'] = entities
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # Store results
        st.session_state.processed_data = {
            'title': title,
            'duration': duration,
            'segments': transcribed_segments,
            'url': youtube_url
        }
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

def display_results():
    """Display the processed results"""
    data = st.session_state.processed_data
    
    # Video info
    st.subheader(f"📺 {data['title']}")
    st.write(f"Duration: {data['duration']:.0f} seconds | Segments: {len(data['segments'])}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Timeline", "📊 Analytics", "📝 Transcript"])
    
    with tab1:
        st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
        timeline_fig = create_interactive_timeline(data['segments'])
        st.plotly_chart(timeline_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        speaker_stats = create_speaker_analytics_dashboard(data['segments'])
        display_speaker_dashboard(speaker_stats)
    
    with tab3:
        st.subheader("📝 Full Transcript")
        
        for i, segment in enumerate(data['segments']):
            with st.expander(f"{segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s)"):
                st.write(segment['text'])
                
                if 'sentiment' in segment:
                    sentiment_color = "green" if segment['sentiment'] == 'Positive' else "red" if segment['sentiment'] == 'Negative' else "gray"
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{segment['sentiment']}</span> ({segment['sentiment_score']:.2f})", unsafe_allow_html=True)
                
                if 'entities' in segment and segment['entities']:
                    st.write("**Entities:**", ", ".join([f"{ent[0]} ({ent[1]})" for ent in segment['entities']]))

if __name__ == "__main__":
    main()
