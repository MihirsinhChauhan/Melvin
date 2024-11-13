from llama_index.core import (
    VectorStoreIndex
)
from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.core.node_parser import TokenTextSplitter
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
# from llama_index.readers.youtube_metadata import YouTubeMetaData
import os
import json
from typing import Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from melvin.utils.utils import format_summary

class YouTubeSummarizer:
    def __init__(
        self,
        model_url = 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf'
        # model_url: str = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        # chunk_size: int = 1024,
        # chunk_overlap: int = 20,
        # api_key: str = "your_youtube_api_key"
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
            model_url=model_url,
            temperature=0.1,
            max_new_tokens=512,
            context_window=4096,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},  # Auto-detect GPU layers
            verbose=True
        )
        
        # # Create service context
        # self.service_context = ServiceContext.from_defaults(
        #     llm=self.llm,
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap
        # )
        
        # Initialize text splitter
        # self.text_splitter = TokenTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap
        # )
        
        # Initialize YouTube transcript and metadata readers
        # self.yt_transcript_reader = YoutubeTranscriptReader()
        # self.yt_metadata_reader = YouTubeMetaData(api_key=api_key)

    # def create_index(self, text: str) -> VectorStoreIndex:
    #     """Create a vector store index from the text"""
    #     # documents = [Document(text=chunk) for chunk in self.text_splitter.split_text(text)]
    #     index = VectorStoreIndex.from_documents(
    #         documents,
            
    #     )
    #     return index
    def create_index(self, video_url: str) -> VectorStoreIndex:
        """Create a vector store index from the text"""
        # documents = [Document(text=chunk) for chunk in self.text_splitter.split_text(text)]
        print("Fetching transcript...")
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(
            ytlinks=[video_url]
        )

        index = VectorStoreIndex.from_documents(
            documents,
            embed_model =HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
            
        )
        return index

    def generate_summary(self, index: VectorStoreIndex) -> dict:
        """Generate comprehensive summary using LlamaIndex"""
        query_engine = index.as_query_engine(llm=self.llm)
        
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

    def process_video(self, video_url: str, save_output: bool = True) -> Optional[dict]:
        """Process YouTube video: fetch transcript and generate comprehensive summary"""
        # print("Fetching transcript...")
        # # transcript = self.yt_transcript_reader.load_data(video_ids)
        # if not transcript:
        #     return None

        print("Creating index...")
        index = self.create_index(video_url)

        print("Generating summary...")
        summary = self.generate_summary(index)
        print(summary)
        
        result = {
            "video url": video_url,
            "summary": summary
        }
        
        # Save results if requested
        if save_output:
            with open("video_summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    

# import os
# import argparse
# from dotenv import load_dotenv
# import getpass

def main():
    # Example usage
    # load_dotenv()

    # api_key = os.environ.get("YOUTUBE_API")
    # video_url = getpass.getpass('Video URL : ')
    # print(video_url)
    # video_id = [get_youtube_id(video_url)]
    # print('vedioID : ',video_id)
    video_url = input("Video URL : ")
    summarizer = YouTubeSummarizer()
    result = summarizer.process_video(video_url)
    
    if result:
        print(format_summary(result["summary"]))
        print("\nFull results saved to video_summary.json")
    else:
        print("Failed to process video")

if __name__ == "__main__":

    # parser = argparse.ArgumentParser("Testing Youtube Summarizer")
    # parser.add_argument('vedio_url', type=str, help= "The link of youtube vedio you want to summarize")

    # args = parser.parse_args()

    main()