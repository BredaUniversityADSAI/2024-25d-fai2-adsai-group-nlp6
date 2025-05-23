# UNIT TESTING

## DIFFERENT FILES

- [Test Transcript](#transcript)
- [Test Data](#data)
- [Test Model](#model)
- [Test Training](#training)
- [Test CLI](#cli)
- [Test API](#api)
## Transcript

### The Transcript class provides functionality to:

Download audio from YouTube videos

Process the audio using either Whisper or AssemblyAI transcription services

Format and save the transcriptions as CSV files

The accompanying test suite (TestTranscript) thoroughly validates all aspects of this functionality.

### Methods Tested
#### __init__ Method

✅ Valid inputs (with various formats)

✅ Invalid inputs

✅ Correct choice storage

#### download_youtube_audio Method

✅ Operation when directory doesn't exist

✅ Operation when directory exists

✅ Exception handling

✅ Return value verification

#### seconds_to_hms Method

✅ Various inputs (zero, small, large numbers)

✅ Decimal input handling

✅ Day boundary (23:59:59)

#### whisper_model Method

✅ File not found scenario

✅ Exception handling during transcription

✅ Full successful functionality

✅ CSV output format

✅ File cleanup

#### transcribe_audio_with_assemblyai Method

✅ Full functionality

✅ API key setup

✅ CSV writing

✅ File cleanup

#### process Method

✅ Whisper choice path

✅ Assembly choice path

✅ Exception handling

## Data

## Model

## Training

## CLI

## API
