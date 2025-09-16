import os
import logging
import tempfile
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader
from pydantic import BaseModel, Field
from typing import List

class ApplicationOutput(BaseModel):
    cover_letter: str = Field(description="Professional cover letter tailored to the job posting")
    adapted_resume: str = Field(description="Resume content adapted to match job requirements")
    key_improvements: List[str] = Field(description="List of key improvements made to the resume")

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
        temperature=0.7
    )
    app.logger.info("Groq LLM initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize Groq LLM: {e}")
    llm = None

try:
    embeddings = None
    app.logger.info("Embeddings setup ready")
except Exception as e:
    app.logger.error(f"Failed to initialize embeddings: {e}")
    embeddings = None

memory = ConversationBufferMemory()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

output_parser = PydanticOutputParser(pydantic_object=ApplicationOutput)

def extract_text_from_pdf(file_path):
    """Extract text content from PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        return None

def allowed_file(filename):
    """Check if the uploaded file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

PROMPT_TEMPLATE = """You are a professional job application assistant. Given a job posting and a candidate's resume, perform the following tasks:

1. Generate a professional cover letter tailored specifically to the job posting
2. Adapt and improve the resume to better match the job requirements while staying truthful to the candidate's experience

Job Posting:
{job_posting}

Current Resume:
{resume}

Please provide your response in the following format:

COVER LETTER:
[Your tailored cover letter here]

ADAPTED RESUME:
[Your improved resume here]

Make sure the cover letter is professional, engaging, and specifically addresses the job requirements. For the resume, reorganize and emphasize relevant skills and experiences while maintaining truthfulness."""

@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle PDF resume upload (Windows-safe temp file handling)"""
    try:
        if 'resume_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['resume_file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(fd)

        try:
            file.save(temp_path)

            resume_text = extract_text_from_pdf(temp_path)

            if not resume_text or not resume_text.strip():
                return jsonify({'error': 'Failed to extract text from PDF'}), 500

            return jsonify({
                'success': True,
                'resume_text': resume_text,
                'filename': secure_filename(file.filename or 'resume.pdf')
            })

        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    except Exception as e:
        app.logger.error(f"Error uploading resume: {e}")
        return jsonify({'error': f'Failed to process file: {e}'}), 500


@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_application():
    """Generate cover letter and adapted resume"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        job_posting = data.get('job_posting', '').strip()
        resume = data.get('resume', '').strip()
        
        if not job_posting or not resume:
            return jsonify({'error': 'Both job posting and resume are required'}), 400
        
        if not llm:
            return jsonify({'error': 'AI service is not available'}), 500
        
        prompt = PROMPT_TEMPLATE.format(
            job_posting=job_posting,
            resume=resume
        )
        
        app.logger.debug(f"Sending prompt to Groq API")
        
        response = llm.invoke([HumanMessage(content=prompt)])
        generated_content = str(response.content) if hasattr(response, 'content') else str(response)
        
        app.logger.debug(f"Received response from Groq API")
        
        cover_letter = ""
        adapted_resume = ""
        
        if "COVER LETTER:" in generated_content and "ADAPTED RESUME:" in generated_content:
            parts = str(generated_content).split("ADAPTED RESUME:")
            cover_letter_part = parts[0].replace("COVER LETTER:", "").strip()
            adapted_resume_part = parts[1].strip() if len(parts) > 1 else ""
            
            cover_letter = cover_letter_part
            adapted_resume = adapted_resume_part
        else:
            lines = str(generated_content).split('\n')
            current_section = None
            cover_letter_lines = []
            resume_lines = []
            
            for line in lines:
                if "COVER LETTER" in line.upper():
                    current_section = "cover_letter"
                    continue
                elif "ADAPTED RESUME" in line.upper() or "RESUME" in line.upper():
                    current_section = "resume"
                    continue
                
                if current_section == "cover_letter":
                    cover_letter_lines.append(line)
                elif current_section == "resume":
                    resume_lines.append(line)
            
            cover_letter = '\n'.join(cover_letter_lines).strip()
            adapted_resume = '\n'.join(resume_lines).strip()
        
        if not cover_letter and not adapted_resume:
            cover_letter = generated_content
            adapted_resume = "Please see the cover letter section for the complete response."
        
        return jsonify({
            'cover_letter': cover_letter,
            'adapted_resume': adapted_resume
        })
        
    except Exception as e:
        app.logger.error(f"Error generating application: {str(e)}")
        return jsonify({'error': f'Failed to generate application: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'llm_available': llm is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
