def transcript():
    while True:
        choice = input("Choose one (whisper/assembly): ").strip().lower()
        if choice in ["whisper", "assembly"]:
            break
        print("Invalid choice. Please enter whisper or assembly.")
    print(f"You chose {choice}!")

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

    if choice == 'whisper':
        print("You chose Whisper")
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
        
    elif choice == 'assembly':
        print("You chose AssemblyAI")
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