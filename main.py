import asyncio
import os
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc

# Load environment variables from .env file
load_dotenv()

async def entrypoint(ctx: JobContext):
    print("Entrypoint started")

    # Load and print the API key for debugging
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    print(f"Using API Key: {OPENAI_API_KEY}")

    # Define the initial context for the LLM
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, avoiding unpronounceable punctuation."
        ),
    )

    # Connect to the LiveKit room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print("Connected to LiveKit room")

    # Create an instance of AssistantFnc
    fnc_ctx = AssistantFnc()

    # Initialize components
    try:
        vad = silero.VAD.load()  # Use Silero VAD
        print("VAD initialized successfully")
    except Exception as e:
        print(f"Failed to initialize VAD: {e}")
        return

    try:
        stt = openai.STT()  # Initialize the STT model
        print("STT initialized successfully")
    except Exception as e:
        print(f"Failed to initialize STT: {e}")
        return

    try:
        llm_model = openai.LLM()  # Initialize the LLM model
        print("LLM initialized successfully")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    try:
        tts = openai.TTS()  # Initialize the TTS model
        print("TTS initialized successfully")
    except Exception as e:
        print(f"Failed to initialize TTS: {e}")
        return

    # Initialize the voice assistant with the required components
    assistant = VoiceAssistant(
        vad=vad,
        stt=stt,
        llm=llm_model,
        tts=tts,
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    assistant.start(ctx.room)
    print("Voice Assistant started")

    await asyncio.sleep(1)  # Allow some time for initialization
    await assistant.say("Hi there, I am Kancha, your very own personal voice assistant", allow_interruptions=True)

    # Continuous interaction loop
    while True:
        await process_audio(assistant, vad, stt, llm_model, tts)

async def process_audio(assistant, vad, stt, llm_model, tts):
    print("Listening for audio input...")
    
    audio_input = await assistant.listen()  # Ensure this is the correct method for listening
    print("Audio input received")

    if audio_input:  # Check if audio input was received
        # Process audio input for VAD
        sample_rate = 16000  # Set the sample rate as per your audio settings
        frame_duration = 30   # Duration of each frame in milliseconds
        frame_length = int(sample_rate * frame_duration / 1000)

        # Split audio input into frames
        frames = [audio_input[i:i + frame_length] for i in range(0, len(audio_input), frame_length)]

        # Check for speech in the frames using VAD
        vad_result = any(vad.is_speech(frame, sample_rate) for frame in frames if len(frame) == frame_length)
        print(f"VAD Result: {vad_result}")

        if vad_result:  # If VAD detected audio
            text_output = await stt.transcribe(audio_input)  # Transcribe audio to text
            print(f"Transcribed Text: {text_output}")

            if text_output:  # Check if transcription is successful
                llm_response = await llm_model.generate(text_output)  # Generate response using LLM
                print(f"LLM Response: {llm_response}")

                await tts.speak(llm_response)  # Convert LLM response to speech
                print("Response spoken")
            else:
                print("No transcribed text received.")
        else:
            print("No audio detected by VAD")
    else:
        print("No audio input received")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))