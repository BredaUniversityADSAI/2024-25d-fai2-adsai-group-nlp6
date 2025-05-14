def download_youtube_audio():
    # importing packages
    from pytubefix import YouTube
    import os

    # url input from youtube
    yt = YouTube("https://www.youtube.com/watch?v=zER2qFdiNp4")

    # extract only audio
    video = yt.streams.filter(only_audio=True).first()

    # set destination to save file
    destination = ("../../data/transcript/")
    if not os.path.exists(destination):
        os.makedirs(destination)

    # download the file
    out_file = video.download(output_path=destination)

    # save the file
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    return new_file

def transcribe_audio_with_assemblyai(new_file):
    import assemblyai as aai
    import csv
    import os
    from datetime import timedelta

    # Basic setup
    aai.settings.api_key = "fb2df8accbcb4f38ba02666862cd6216"

    # Setup paths
    audio_path = new_file
    output_path = os.path.join('..', '..', 'data', 'transcript', 'transcript.csv')
    print("Starting transcription...")

    # Create transcriber and process file
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)
    def format_time(seconds):
        td = timedelta(seconds=seconds)
        # Format as H:MM:SS (removes microseconds)
        return str(td).split('.')[0]
    # Save to CSV in Scripts folder
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Sentence Number", "Text", "Start Time", "End Time"])
        
        for i, sentence in enumerate(transcript.get_sentences(), 1):
            start_time_sec = round(sentence.start / 1000, 2)
            end_time_sec = round(sentence.end / 1000, 2)
            start_time_str = format_time(start_time_sec)
            end_time_str = format_time(end_time_sec)
            writer.writerow([i, sentence.text, start_time_str, end_time_str])
            print(f"Sentence {i} saved")
    print(f"Done! Check {output_path} for the output.")
    os.remove(new_file)
    return new_file

new_file = download_youtube_audio()
transcribe_audio_with_assemblyai(new_file)