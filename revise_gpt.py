from dotenv import load_dotenv
import os
import argparse
from openai import AzureOpenAI
import time
# Load environment variables from the .env file
load_dotenv()

def load_transcript(file_path):
    """
    Load the transcript from a text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return None

def generate_minutes(transcript):
    """
    Generate meeting minutes from the transcript using Azure OpenAI.
    Returns the generated minutes appended with metadata (e.g., token usage).
    """
    # System message defining your role as a professional meeting minutes transcriber
    system_message1 = """
    You are an expert meeting minutes transcriber. You will be provided with a text file containing a transcription of a recorded meeting. Although the transcription is highly accurate, some sections may be unclear due to stammering, hesitations, or difficult-to-hear audio. Your task is to review, edit, and transform the transcription into comprehensive, clear, and official meeting minutes that are accurate, easy to understand, and well-balanced in detail. Make sure that any questions raised during the meeting and their corresponding answers are clearly presented.
    """
    system_message2 = """
    Please adhere strictly to the following guidelines:
    - Preserve the original timestamps and speaker identifiers (e.g., SPEAKER_0, SPEAKER_1) as much as possible.
    - Since most meetings will be in Japanese, your final output must be in Japanese.
    - Your meeting minutes must include the following items without any omissions:
    • 参加者 (Participants): List all participants exactly as indicated by the speaker identifiers in the transcript (e.g., SPEAKER_0, SPEAKER_1, SPEAKER_2, ...).
    • 決定事項 (Decisions Made): List all decisions made during the meeting, ensuring that nothing is omitted.
    • 次に取るべき行動 (Next Actions): List all follow-up actions to be taken after the meeting.
    • 主な会議の内容 (Key Discussion Points): Thoroughly list all the main topics discussed during the meeting. For each topic, provide a brief description of the context, the key arguments presented, and any supporting details that clarify the discussion.
        - A brief description of the context and background.
        - The key arguments presented, including supporting details that clarify the discussion.
        - Details of significant Q&A exchanges (noting who said what) that influenced the discussion or decisions.
• 現状・理想・手段 (Current Situation, Ideal State, Means): Create a dedicated section that clearly and concisely outlines three critical aspects:
    - **Current Situation**: Summarize the present state of affairs, highlighting existing challenges, limitations, or conditions discussed during the meeting. Emphasize any key issues or constraints that are impacting the current operations.
    - **Ideal State**: Define the desired or optimal state that the team or organization aims to achieve. This should include the envisioned improvements or outcomes that address the current challenges and set a clear, attainable goal.
    - **Means**: Detail the proposed methods, strategies, or actionable steps intended to transition from the current situation to the ideal state. Specify concrete initiatives, timelines, or processes that were suggested to achieve the targeted improvements.
    
    """
    # User message including the transcript content
    user_message = f"This is the text of the audio transcription of the meeting. Make sure they are appropriate for the minutes.\n\n{transcript}"

    try:
        start = time.perf_counter()
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = "2024-10-21"  # Specify the Azure OpenAI API version
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message1},
                {"role": "system", "content": system_message2},
                {"role": "user", "content": user_message},
            ],
            max_tokens=4096,      # Adjust according to the desired length of the meeting minutes
            temperature=0.5,      # Lower value for more deterministic output
        )
        # Extract the generated meeting minutes from the response
        output = response.choices[0].message.content

        # Extract token usage metadata if available
        usage = response.usage
        total_tokens = usage.total_tokens
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        # Prepare metadata text
        end = time.perf_counter()
        elapsed = end - start
        metadata = (
            "\n\n--- Metadata ---\n"
            f"Total tokens: {total_tokens}\n"
            f"Prompt tokens: {prompt_tokens}\n"
            f"Completion tokens: {completion_tokens}\n"
            f"Elapsed time: {elapsed:.2f} seconds\n"
        )
        # Append metadata to the generated minutes
        return output + metadata
    except Exception as e:
        print("Error generating minutes:", e)
        return None

def main(args):
    # Load environment variables
    load_dotenv()
    
    transcript_file = args.text_path
    transcript = load_transcript(transcript_file)
    if not transcript:
        print("Error: Transcript file not found or could not be loaded.")
        return
    
    # Generate meeting minutes along with metadata
    
    minutes = generate_minutes(transcript)
    if minutes:
        print("Generated meeting minutes:")
        print(minutes)
        # Write the minutes and metadata to the output file
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(minutes)
    else:
        print("Error: Minutes generation failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe and generate meeting minutes using Azure OpenAI"
    )
    parser.add_argument(
        "--text_path", type=str, required=True, help="Path to the input transcript file"
    )
    parser.add_argument(
        "--output_path", type=str, default="./meeting_minutes.txt", help="Path to save the output meeting minutes"
    )
    args = parser.parse_args()
    main(args)
