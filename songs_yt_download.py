import os
import subprocess

output_dir = "Tamer_Hosny_Songs"
songs_file = "song names.txt"  
#Format in the text file should be like this:
# 1. Tamer Hosny - Bahebak

os.makedirs(output_dir, exist_ok=True)


with open(songs_file, 'r', encoding='utf-8') as f:
    for line in f:
        query = line.strip()  
        if not query: 
            continue
        print(f"Searching and downloading: {query}")
        try:
            subprocess.run([
                "yt-dlp",
                f"ytsearch1:{query}",
                "--extract-audio",
                "--audio-format", "mp3",
                "-o", os.path.join(output_dir, "%(title)s.%(ext)s")
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download: {query} - Error: {e}")
        except FileNotFoundError:
            print("Error: yt-dlp not found. Make sure it's installed and in your PATH.")
            break 