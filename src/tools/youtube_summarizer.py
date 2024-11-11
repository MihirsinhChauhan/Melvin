from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    Document
)
from llama_index.llms import LlamaCPP
from llama_index.text_splitter import TokenTextSplitter
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.youtube_metadata import YoutubeMetaData
import os
import json
from typing import Optional

class YouTubeSummarizer:
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        api_key: str = "your_youtube_api_key"
    ):
        """
        Initialize the YouTube Summarizer
        
        Args:
            model_path: Path to the local LLM model file
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            api_key: Your YouTube API key
        """
        # Initialize LLM
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=512,
            context_window=4096,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},  # Auto-detect GPU layers
            verbose=True
        )
        
        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize text splitter
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize YouTube transcript and metadata readers
        self.yt_transcript_reader = YoutubeTranscriptReader(api_key=api_key)
        self.yt_metadata_reader = YoutubeMetaData(api_key=api_key)

    def create_index(self, text: str) -> VectorStoreIndex:
        """Create a vector store index from the text"""
        documents = [Document(text=chunk) for chunk in self.text_splitter.split_text(text)]
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        return index

    def generate_summary(self, index: VectorStoreIndex) -> dict:
        """Generate comprehensive summary using LlamaIndex"""
        query_engine = index.as_query_engine()
        
        # Define prompts for different aspects of the summary
        prompts = {
            "main_points": "What are the main points discussed in this video?",
            "key_insights": "What are the key insights or takeaways?",
            "detailed_summary": "Provide a detailed summary of the content.",
            "timeline": "Create a timeline of main topics discussed in chronological order."
        }
        
        summary = {}
        for aspect, prompt in prompts.items():
            try:
                response = query_engine.query(prompt)
                summary[aspect] = str(response)
            except Exception as e:
                print(f"Error generating {aspect}: {str(e)}")
                summary[aspect] = f"Error generating {aspect}"
        
        return summary

    def process_video(self, video_ids: list, save_output: bool = True) -> Optional[dict]:
        """Process YouTube video: fetch transcript and generate comprehensive summary"""
        print("Fetching transcript...")
        transcript = self.yt_transcript_reader.load_data(video_ids)
        if not transcript:
            return None

        print("Creating index...")
        index = self.create_index(transcript)

        print("Generating summary...")
        summary = self.generate_summary(index)
        
        result = {
            "video_ids": video_ids,
            "transcript": transcript,
            "summary": summary
        }
        
        # Save results if requested
        if save_output:
            with open("video_summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result