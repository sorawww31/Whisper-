# Whisper & Pyannote - Test

This repository is used to evaluate a product that combines **speaker diarization** and **speech transcription** using [Whisper](https://github.com/openai/whisper) and [Pyannote](https://github.com/pyannote/pyannote-audio).

## 🛠️ Setup

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

> **Recommended Python version:** `3.11.11`

## 🎙️ Transcription with Faster-Whisper

This project uses various `faster-whisper` models from [Hugging Face](https://huggingface.co/) to perform audio transcription.

You can specify the model using the `--model_size` option. Example model options include:

- `deepdml/faster-whisper-large-v3-turbo-ct2`
- `mobiuslabsgmbh/faster-whisper-large-v3-turbo`
- `Systran/faster-distil-whisper-large-v3`

## 🔊 Input Options

Use the `--wav_file` option to specify the WAV file you want to transcribe.

Alternatively, you can use the `--input_file` option to provide an MP4 video file. It will automatically be converted to WAV format and used as input for transcription.

## 🚀 Example Usage

```bash
python main.py \
  --model_size deepdml/faster-whisper-large-v3-turbo-ct2 \
  --wav_file /data/test.wav \
  --num_speakers 2 \
  --output_file outpus/fast-v3-turbo-ct2.txt \
  --device cuda
```

## 📌 Notes

- Make sure the specified model exists on Hugging Face and is compatible with `faster-whisper`.
- For better performance, using a GPU (`--device cuda`) is recommended.
- You **must have a Hugging Face API token** to download the models.  
  You can obtain one from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- It's recommended to store your token securely in a `.env` file like this:

  ```
  HUGGINGFACE_TOKEN=your_token_here
  ```

  Then load it in your script using `python-dotenv` or a similar method.
