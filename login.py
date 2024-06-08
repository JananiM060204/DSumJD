import pandas as pd
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import spacy
from fpdf import FPDF
import os
import tempfile

# Load the BART model and tokenizer only once
@st.cache_resource
def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

tokenizer, model = load_bart_model()

@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        return spacy.load('en_core_web_sm')
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "spacy==3.1.3"])
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Load the EHR data from the Excel file
@st.cache_data  # Cache the data to avoid loading it multiple times during the session
def load_data():
    df = pd.read_excel("sample_ehr_data_Jan.xlsx")
    return df

# Function to generate a summary using BART
def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate a discharge summary
def generate_discharge_summary(patient):
    case_history = patient['Case History'].values[0]
    doc = nlp(case_history)
    sentences = [sent.text for sent in doc.sents]

    # Combine case history with report document summaries
    if not pd.isna(patient['Report_Document'].values[0]):
        report_summary = " Report Summary: Detailed report information."
        # Here you might extract text from the PDF and summarize it as well, if needed.
    else:
        report_summary = ""

    # Limit the input text length for summary generation
    max_sentences = 10
    limited_text = ' '.join(sentences[:max_sentences])
    full_text = f"{limited_text} {report_summary}"
    discharge_summary = generate_summary(full_text)
    return discharge_summary

def create_pdf(hospital_name, patient_name, age, gender, date_of_admission, discharge_summary, follow_up_treatment, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Add the Maven Pro font
    pdf.add_font("MavenPro-Regular", fname="static/Maven_Pro/MavenPro-VariableFont_wght.ttf", uni=True)
    pdf.set_font("MavenPro-Regular", size=12)

    # Add background image
    pdf.image("static/Dbg.png", x=0, y=0, w=210, h=297, type='', link='')

    # Display hospital name in bold caps at the center
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(0, 10, hospital_name.upper(), ln=True, align="C")

    # Set font back to normal
    pdf.set_font("Arial", size=12)

    # Display patient's details
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Gender: {gender}", ln=True)
    pdf.cell(0, 10, f"Date of Admission: {date_of_admission}", ln=True)
    pdf.cell(0, 10, "", ln=True)  # Empty line for spacing

    # Display discharge summary
    pdf.cell(0, 10, "Discharge Summary:", ln=True)
    pdf.multi_cell(0, 10, discharge_summary)
    pdf.cell(0, 10, "", ln=True)  # Empty line for spacing

    # Display follow-up treatment
    pdf.cell(0, 10, "Follow-up Treatment:", ln=True)
    pdf.multi_cell(0, 10, follow_up_treatment)

    pdf.output(filename)

def main():
    st.title("Patient Portal")

    # Load the data when the app starts
    df = load_data()

    # Initialize session state variables
    if 'login_successful' not in st.session_state:
        st.session_state.login_successful = False
    if 'discharge_summary' not in st.session_state:
        st.session_state.discharge_summary = None
    if 'pdf_filename' not in st.session_state:
        st.session_state.pdf_filename = None
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None

    # Sidebar for admin login
    st.sidebar.subheader("Admin Login")
    patient_id = st.sidebar.text_input("Enter Patient ID:")
    patient_name = st.sidebar.text_input("Enter Patient Name:")
    login_button = st.sidebar.button("Login")

    if login_button:
        try:
            # Filter the DataFrame based on patient ID and name
            patient = df[(df["Patient_ID"] == int(patient_id)) & (df["Patient_Name"].str.lower() == patient_name.lower())]

            if not patient.empty:
                st.session_state.login_successful = True
                st.session_state.patient_id = patient_id
                st.session_state.patient_name = patient_name
                st.session_state.patient_data = patient
                st.success("Login successful!")
            else:
                st.error("Patient ID and/or Name is incorrect. Please try again.")
        except ValueError:
            st.error("Invalid Patient ID. Please enter a numeric ID.")

    if st.session_state.login_successful:
        patient_id = st.session_state.patient_id
        patient_name = st.session_state.patient_name
        patient = st.session_state.patient_data

        # Display detailed patient history in a formatted way
        st.subheader(f"Detailed History for Patient: {patient_name} (ID: {patient_id})")
        for index, row in patient.iterrows():
            st.write(f"Date of Admission: {row['Date_of_Admission']}")
            st.write(f"Admitting Doctor: {row['Handled Doctor']}")
            st.write(f"Discharge Date: {row['Date of Discharge']}")
            # Display Case History in a formatted manner
            case_history = row['Case History'].split('\n')
            st.write("Case History:")
            for entry in case_history:
                st.write(f"- {entry}")

            # Display Report Document if available
            if not pd.isna(row['Report_Document']):
                # Get the URL from the 'Report_Document' column in your DataFrame
                pdf_url = row['Report_Document']
                # Create a clickable and downloadable link
                st.markdown(f"[View Reports]({pdf_url})")
        
        # Add the generate discharge summary button in the sidebar
        if st.sidebar.button("Generate Discharge Summary"):
            if patient is not None:
                discharge_summary = generate_discharge_summary(patient)
                st.session_state.discharge_summary = discharge_summary
                
                # Create a PDF and save it to a temporary file
                hospital_name = "Hospital XYZ"  # Replace with actual data from your Excel sheet
                age = patient['Age'].values[0]  # Example age from DataFrame
                gender = patient['Gender'].values[0]  # Example gender from DataFrame
                date_of_admission = patient['Date_of_Admission'].values[0]  # Example date from DataFrame
                follow_up_treatment = "Follow-up appointments scheduled for next week."  # Example treatment, replace with actual data

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    st.session_state.pdf_filename = tmp_file.name
                    create_pdf(hospital_name, patient_name, age, gender, date_of_admission, discharge_summary, follow_up_treatment, tmp_file.name)
        
        if st.session_state.pdf_filename:
            # Provide view link for the discharge summary PDF
            with open(st.session_state.pdf_filename, "rb") as f:
                st.sidebar.download_button(
                    label="View Discharge Summary",
                    data=f,
                    file_name=f"Discharge_Summary_{patient_id}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()

