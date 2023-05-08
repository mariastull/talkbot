# Talkbot

Actually chat with a chatbot! It will verbally talk to you, you can talk back and it will respond.

How to use: 
1. Install all requirements (requirements.txt)
2. run talkbot.py
3. Have a conversation!


Talkbot uses openai's [Whisper] (https://openai.com/research/whisper) to locally transcribe speech as it hears it. The live-transcription modification is taken from [davabase's whisper real time repo](https://github.com/davabase/whisper_real_time).

A prompt including the transcription is sent to an [openai gpt model] (https://platform.openai.com/docs/models/overview), which responds.

[gtts](https://pypi.org/project/gTTS/) reads the AI response out loud.

## Notes
Modify the prompt and AI's first words to say whatever you want.

This is also easily modifiable to work in other languages.
