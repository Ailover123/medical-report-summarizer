import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import io
from datetime import datetime
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Medical Report Summarizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    /* Remove default padding */
    .main > div {
        padding-top: 0;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1rem;
    }
    
    /* Divider */
    .divider {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.02);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
        border-left: 4px solid #22c55e;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Premium button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-bottom: 2px solid transparent;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #667eea;
        border-bottom-color: #667eea;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        background: rgba(102, 126, 234, 0.02);
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric boxes */
    [data-testid="metric-container"] {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# --- Functions ---
def get_text_from_pdf(file):
    """Extracts text from an uploaded PDF file with validation."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        page_count = len(pdf_reader.pages)
        
        if page_count == 0:
            st.warning("⚠️ The PDF file appears to be empty.")
            return None
        
        for idx, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            st.warning("⚠️ Could not extract text from the PDF. The file might be image-based.")
            return None
        
        return text
    except Exception as e:
        st.error(f"⚠️ Error reading PDF file: {str(e)}")
        return None

def validate_report_text(text):
    """Validates if the report text is suitable for processing."""
    if not text or len(text.strip()) < 50:
        return False, "Report text must be at least 50 characters long."
    if len(text) > 50000:
        return False, "Report text is too long (max 50,000 characters). Please upload a shorter report."
    return True, "Valid"

def get_gemini_summary(report_text):
    """Generates a summary using the Gemini API with improved prompting."""
    is_valid, message = validate_report_text(report_text)
    if not is_valid:
        st.error(f"⚠️ {message}")
        return None
    
    prompt = f"""
You are an expert medical summarization assistant specializing in patient-friendly explanations.

Your task is to create a clear, compassionate summary of the following medical report for a patient without medical background.
Use simple, everyday language and avoid jargon.

**IMPORTANT INSTRUCTIONS:**
- Replace all medical jargon with plain language explanations
- Use bullet points for key findings (max 5-7 points)
- Keep explanations concise but thorough
- Be empathetic and reassuring in tone
- If anything is unclear or concerning, recommend consulting the healthcare provider

Please structure your response EXACTLY as follows:

---
## 📋 Key Findings Summary
*Brief overview of what the report shows (2-3 sentences)*

## 📌 Main Test Results
*Present findings as bullet points with plain language explanations*
- Result 1: Plain language explanation
- Result 2: Plain language explanation
*(continue as needed)*

## 🩺 What This Means for You
*Explain in simple terms what these findings mean for the patient's health*

## 📅 Recommended Next Steps
*Suggest practical follow-up actions such as:*
- Schedule a follow-up appointment
- Discuss results with your doctor
- Lifestyle considerations if applicable
*(Do NOT provide medical advice or prescribe treatments)*

## ⚠️ Important Disclaimer
*This is an AI-generated summary for informational purposes only. It is NOT a substitute for professional medical advice. Always discuss your results with your healthcare provider. In case of emergency, contact emergency services immediately.*

---

**Medical Report to Summarize:**
{report_text}
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"⚠️ Error generating summary: {str(e)}")
        return None

def save_summary_to_history(filename, summary, char_count):
    """Saves summary to session history."""
    st.session_state.summary_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "summary": summary,
        "char_count": char_count
    })

def export_summary_as_text(summary, filename="medical_summary"):
    """Creates a downloadable text file of the summary."""
    return summary.encode("utf-8")

# --- Authentication and API Key Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🔐 Gemini API Key Not Found")
    st.markdown("""
    <div class="info-box">
    <strong>Setup Required:</strong><br>
    1. Create a <code>.env</code> file in your project directory<br>
    2. Add: <code>GEMINI_API_KEY=your_api_key_here</code><br>
    3. Restart the Streamlit app<br><br>
    <a href="https://makersuite.google.com/app/apikey" target="_blank">Get your API key from Google AI Studio</a>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"⚠️ Failed to configure Gemini API: {str(e)}")
    st.stop()

# --- Header Section ---
st.markdown("""
<div class="header">
    <h1>🏥 Medical Report Summarizer</h1>
    <p>Transform complex medical reports into simple, understandable summaries powered by AI</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("📖 About This Tool")
    st.info(
        "This application uses Google's Gemini AI to convert complex medical terminology into "
        "plain language summaries.\n\n"
        "**Features:**\n"
        "✅ Support for PDF and text files\n"
        "✅ Patient-friendly explanations\n"
        "✅ Structured summaries\n"
        "✅ Download results"
    )
    
    st.divider()
    
    with st.expander("💡 How to Use", expanded=True):
        st.markdown("""
        **Step 1:** Choose your input method
        - Upload a PDF or text file
        - Or paste text directly
        
        **Step 2:** Review the content
        - Make sure your medical report is complete
        
        **Step 3:** Generate summary
        - Click the summarize button
        
        **Step 4:** Download or share
        - Save the summary for your records
        """)
    
    with st.expander("📋 Supported File Types"):
        st.markdown("""
        - **PDF Files** (.pdf)
        - **Text Files** (.txt)
        - **Direct Paste** (copy-paste)
        """)
    
    st.divider()
    
    # History Section
    if st.session_state.summary_history:
        st.subheader("📚 Recent Summaries")
        for idx, item in enumerate(reversed(st.session_state.summary_history[-5:])):
            with st.expander(f"{item['timestamp']} - {item['filename'][:30]}"):
                st.caption(f"Characters: {item['char_count']}")

# --- Main Content ---
# Statistics Row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Summaries Generated", len(st.session_state.summary_history))

with col2:
    st.metric("📁 Supported Formats", "PDF & TXT")

with col3:
    st.metric("⚡ Max File Size", "20 MB")

st.divider()

# --- Input Section ---
st.subheader("📤 Upload Your Medical Report")

input_tab1, input_tab2 = st.tabs(["📄 Upload File", "✍️ Paste Text"])

report_text = ""
filename = "report"

with input_tab1:
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["txt", "pdf"],
        help="Upload your medical report. Max 20MB"
    )
    
    if uploaded_file:
        filename = uploaded_file.name
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if file_size_mb > 20:
            st.error("❌ File is too large. Maximum size is 20MB.")
        else:
            with st.spinner("📖 Reading file..."):
                time.sleep(0.3)
                if uploaded_file.type == "text/plain":
                    try:
                        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                        report_text = stringio.read()
                        st.success(f"✅ Loaded: {len(report_text)} characters")
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                elif uploaded_file.type == "application/pdf":
                    report_text = get_text_from_pdf(uploaded_file)
                    if report_text:
                        st.success(f"✅ Loaded: {len(report_text)} characters")

with input_tab2:
    pasted_text = st.text_area(
        "Paste your medical report here",
        height=300,
        placeholder="Paste the full text of your medical report..."
    )
    
    if pasted_text:
        report_text = pasted_text
        filename = "pasted_report"
        st.info(f"📊 {len(pasted_text)} characters detected")

# --- Report Preview ---
if report_text:
    with st.expander("👁️ Preview Report"):
        st.text(report_text[:300] + "..." if len(report_text) > 300 else report_text)

st.divider()

# --- Generate Button ---
st.subheader("⚡ Generate Summary")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    generate_btn = st.button("🔍 Generate Summary", use_container_width=True, type="primary")

with col_btn2:
    if report_text:
        is_valid, msg = validate_report_text(report_text)
        if is_valid:
            st.success("✅ Ready")
        else:
            st.warning("⚠️ Not ready")

# --- Generate Summary ---
if generate_btn:
    if not report_text:
        st.warning("⚠️ Please upload a file or paste text to generate a summary.")
    else:
        with st.spinner("🤖 AI is analyzing your report... This may take a moment."):
            summary = get_gemini_summary(report_text)
            
            if summary:
                st.session_state.file_processed = True
                save_summary_to_history(filename, summary, len(report_text))
                
                st.divider()
                st.subheader("📋 Your Simplified Medical Report")
                
                st.markdown("""
                <div class="info-box">
                """, unsafe_allow_html=True)
                
                st.markdown(summary)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download Options
                st.divider()
                st.subheader("📥 Download Your Summary")
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    summary_text = export_summary_as_text(summary, filename)
                    st.download_button(
                        label="📄 Download as Text",
                        data=summary_text,
                        file_name=f"{filename.replace('.pdf', '').replace('.txt', '')}_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_d2:
                    st.info("💡 Share this with your healthcare provider")

elif report_text and not st.session_state.file_processed:
    st.info("📌 Click 'Generate Summary' to create your simplified report")

else:
    st.info("📌 Upload a medical report or paste text to begin")

# --- Footer ---
st.divider()
st.markdown("""
<div class="info-box" style="text-align: center; margin-top: 3rem;">
    <strong>⚕️ Medical Disclaimer</strong>
    <br><br>
    This tool provides AI-generated summaries for informational purposes only. It is <strong>NOT</strong> a substitute for professional medical advice. 
    Always discuss your results with your healthcare provider. In case of emergency, contact emergency services immediately.
    <br><br>
    <small>Powered by Google Gemini AI | 2025</small>
</div>
""", unsafe_allow_html=True)