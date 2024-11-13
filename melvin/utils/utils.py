from pytube import extract

def get_youtube_id(url):
    return extract.video_id(url)


def format_summary(summary: dict) -> str:
    """Format the summary for display"""
    formatted = "\n\n=== SUMMARY ===\n\n"
    
    for section, content in summary.items():
        formatted += f"\n{section.replace('_', ' ').upper()}:\n"
        formatted += "=" * 40 + "\n"
        formatted += f"{content}\n"
        formatted += "-" * 40 + "\n"
    
    return formatted