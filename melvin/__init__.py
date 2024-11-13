"""
A Personal Assistantant that will help to manage your life and boost your productivity
"""

__app_name__ = 'melvin'
__version__ ='0.1.0'

from .tools.youtube_summarizer import YouTubeSummarizer


__all__ = [
    'YouTubeSummarizer'
]