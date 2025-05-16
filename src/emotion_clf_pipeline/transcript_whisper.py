
def download_youtube_audio():
    # importing packages
    from pytubefix import YouTube
    import os

    # url input from youtube
    yt = YouTube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # extract only audio
    video = yt.streams.filter(only_audio=True).first()

    # set destination to save file
    destination = os.path.join("data", "transcript")
    if not os.path.exists(destination):
        os.makedirs(destination)

    # download the file
    out_file = video.download(output_path=destination)

    # save the file
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    return os.path.basename(new_file)

def seconds_to_hms(seconds):
    from datetime import timedelta
    return str(timedelta(seconds=int(seconds)))

def whisper_model(audio_file_path):
    import os
    import whisper
    import csv

    model_type = "medium"
    output_dir = os.path.join('data', 'transcript')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Join the output directory with the audio file path
    audio_path = os.path.join(output_dir, audio_file_path)
        
    output_path = os.path.join(output_dir, 'transcript.csv')

    # Load the Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model(model_type)
    print(audio_path)
    
    # Verify the file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    print(f"Transcribing file: {audio_path}")
    
    # Transcribe the MP3 file
    try:
        result = model.transcribe(audio_path)
        # ...existing code...
    
        # Save the transcription as a CSV file
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["Start (HH:MM:SS)", "End (HH:MM:SS)", "Sentence"])

            # Write transcription text (split into sentences)
            for segment in result["segments"]:
                start_hms = seconds_to_hms(segment["start"])
                end_hms = seconds_to_hms(segment["end"])
                writer.writerow([start_hms, end_hms, segment["text"].strip()])

        print(f"Transcription saved to {output_path}")
        os.remove(audio_path)
    except Exception as e:
        print(f"Transcription error: {e}")

# Run the functions
try:
    new_file = download_youtube_audio()
    print(f"Downloaded audio file: {new_file}")
    whisper_model(new_file)
except Exception as e:
    print(f"Error in pipeline: {e}")