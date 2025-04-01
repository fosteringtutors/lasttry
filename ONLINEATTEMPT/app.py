from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import openai
import re
from datetime import datetime  # For timestamped filenames


UPLOAD_FOLDER = "recordings"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists

app = Flask(__name__)

MOCKS_FOLDER = "mocks"

# Load API Key from environment variables
def load_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    return None

# Load AssemblyAI API Key from environment variables
def load_assemblyai_key():
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if api_key:
        return api_key
    return None
    
def get_latest_transcription(mock_name, question_num):
    # Directory where transcriptions are saved
    mock_dir = os.path.join(UPLOAD_FOLDER, mock_name)
    question_dir = os.path.join(mock_dir, f"question_{question_num}")
    
    if not os.path.exists(question_dir):
        return None  # No transcriptions found
    
    # List all transcription files and sort by timestamp
    transcription_files = sorted(
        [f for f in os.listdir(question_dir) if f.startswith("transcript_") and f.endswith(".txt")],
        reverse=True  # Sort in descending order to get the most recent first
    )
    
    if transcription_files:
        # Return the path of the most recent transcription
        latest_transcription = transcription_files[0]
        transcription_path = os.path.join(question_dir, latest_transcription)
        
        # Read the transcription text from the file
        with open(transcription_path, "r") as file:
            transcription_text = file.read()
        
        return transcription_text
    
    return None  # No transcription found


# Regex to Extract Score
def extract_score(feedback):
    match = re.search(r'(\d+)/10', feedback)
    if match:
        return int(match.group(1))
    
    # If no score is found, return 0 (fallback case)
    return 0

# Load mock questions
def get_mock_files():
    """Retrieve available mock interview files."""
    if not os.path.exists(MOCKS_FOLDER):
        return []
    return sorted([f for f in os.listdir(MOCKS_FOLDER) if f.startswith("mock") and f.endswith(".json")])

def load_mock_questions(mock_name):
    """Load questions from a selected mock JSON file."""
    mock_path = os.path.join(MOCKS_FOLDER, mock_name)
    if not os.path.exists(mock_path):
        return []
    with open(mock_path, "r") as file:
        return json.load(file)

# Core Evaluation Function
def evaluate_response(question_block, user_answer):
    try:
        question = question_block["question"]
        good_points = ", ".join(question_block["mark_scheme"]["good_points"])
        red_flags = ", ".join(question_block["mark_scheme"]["red_flags"])

        mark_scheme = f"""
        **Good Points (+1)**: {good_points}
        **Red Flags (-2)**: {red_flags}

        Scoring System:
        - 4/10 minimum for what sounds like a reasonably fine answer.
        - If they don’t answer the question, cap score at 0.
        - If they are offensive, cap score at 0.
        - Good points +1 each.
        - if less than 280 words are submitted, cap at 3.
        - Red flags -2 each.
        - Cap score between 0-10.
        """

        prompt = f"""
        You are an AI trained to score UK medical school interview answer transcriptions.

        Question: {question}
        Candidate's Response: {user_answer}
        Mark Scheme: {mark_scheme}

        Evaluate the candidate's answer, give a score out of 10 based on the scoring system and mark scheme, and explain what they did well and what they should do differently according to the mark scheme , say they 'could' do this rather than they didnt. do not directly refer to the mark scheme or word count. Make sure they directly mention the green flags, dont just assume they have hit it. do not give use words that are hyphenated. 

        Format your response as follows:
        **Score: X/10**  

        **What you did well:**  
        - [List each green flag they mention on a new line]  

        **How you could improve:**  
        - [List which good points they missed and which red flags they hit, each on a new line]  

        Ensure the response is structured neatly, with each point formatted as a bullet.
        """

        api_key = load_api_key()
        if not api_key:
            return {"error": "API Key not found."}, 500

        openai.api_key = api_key

        ai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical school interview scorer."},
                {"role": "user", "content": prompt}
            ]
        )

        evaluation = ai_response['choices'][0]['message']['content']
        score = extract_score(evaluation)

        return {"score": score, "feedback": evaluation}

    except KeyError as e:
        return {"error": f"Missing data: {str(e)}"}, 400

    except openai.error.OpenAIError as e:
        return {"error": str(e)}, 500


# Routes
@app.route("/")
def index():
    """Home screen listing available mocks."""
    mock_files = get_mock_files()
    return render_template("index.html", mocks=mock_files)

@app.route("/mock/<mock_name>/question/<int:question_num>", methods=["GET", "POST"])
def show_question(mock_name, question_num):
    # Load the questions and mock data
    questions = load_mock_questions(mock_name)
    
    if 1 <= question_num <= len(questions):
        question_data = questions[question_num - 1]
        
        # Get the latest transcription if available
        transcription = get_latest_transcription(mock_name, question_num)

        if request.method == "POST":
            user_answer = request.form.get("answer", "").strip()
            
            if not user_answer:
                return jsonify({"error": "Answer cannot be empty"}), 400
            
            evaluation_data = evaluate_response(question_data, user_answer)
            
            return jsonify(evaluation_data)  # Always return JSON for AJAX requests
        
        return render_template("mock.html", mock_name=mock_name, question=question_data, 
                               question_num=question_num, total=len(questions), transcription=transcription)

    return redirect(url_for("index"))


import assemblyai as aai
import time

ASSEMBLYAI_API_KEY = load_assemblyai_key()
if not ASSEMBLYAI_API_KEY:
    raise ValueError("AssemblyAI API key not found in env.txt")

aai.settings.api_key = ASSEMBLYAI_API_KEY

@app.route("/mock/<mock_name>/question/<int:question_num>/save_audio", methods=["POST"])
def save_audio(mock_name, question_num):
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Invalid file"}), 400

    # Create a structured directory for saving
    mock_dir = os.path.join(UPLOAD_FOLDER, mock_name)
    question_dir = os.path.join(mock_dir, f"question_{question_num}")
    os.makedirs(question_dir, exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(question_dir, f"response_{timestamp}.wav")

    # Save the file locally
    audio_file.save(file_path)

    # Upload to AssemblyAI for transcription
    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(file_path)
    except Exception as e:
        return jsonify({"error": f"AssemblyAI error: {str(e)}"}), 500

    # Save transcription in the same directory
    transcript_path = os.path.join(question_dir, f"transcript_{timestamp}.txt")
    with open(transcript_path, "w") as f:
        f.write(transcript.text)

    return jsonify({
        "message": "Audio saved and transcribed successfully",
        "file_path": file_path,
        "transcription_path": transcript_path,
        "transcription": transcript.text
    }) 

if __name__ == "__main__":
    app.run(debug=True)

