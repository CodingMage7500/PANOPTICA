import streamlit as st
from PIL import Image
import os
import cv2

# Import the logic cleanly from your main file
from backend import (
    load_models, predict, GradCAMSave,
    Initial_translate, AMD_translate, DME_translate
)

# --- DIRECTORY SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# Create folders if they accidentally get deleted
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- SAFE LOGO LOADING & PAGE SETUP ---
# Streamlit requires set_page_config to be the very first st. command!
logo_path = os.path.join(BASE_DIR, "logo.png")

try:
    icon_image = Image.open(logo_path)
    st.set_page_config(page_title="Panoptica", layout="wide", page_icon=icon_image)
except FileNotFoundError:
    # Fallback to the emoji if logo.png isn't found
    st.set_page_config(page_title="Panoptica", layout="wide", page_icon="👁️")
    icon_image = None

# --- UI HEADER ---
st.title("Panoptica: AI Retinal Diagnosis Assistant")

# --- SMART MODEL LOADING ---
@st.cache_resource
def setup_system():
    return load_models()

INITIAL, AMD, DME = setup_system()

# --- SIDEBAR UI: SYSTEM STATUS ---
if icon_image:
    st.sidebar.image(icon_image, use_container_width=True)

st.sidebar.header("System Status")
st.sidebar.success("Main EfficientNet Loaded")
st.sidebar.success("AMD Sub-model Loaded")
st.sidebar.success("DME Sub-model Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Confidence Key:
* **>90%:** Extremely sure
* **80%-90%:** Very sure
* **70%-80%:** Moderately confident
* **<70%:** Unsure
""")

# --- SIDEBAR UI: OUTPUT ARCHIVE VIEWER ---
st.sidebar.markdown("---")
st.sidebar.header("Saved Output Archive")

# Get a list of all images AND text files currently in the Outputs folder
output_files = [f for f in os.listdir(OUTPUTS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.txt'))]
# Sort files so the newest ones appear nicely
output_files.sort(reverse=True)

if not output_files:
    st.sidebar.info("No records saved yet. Run an analysis to see them here!")
else:
    # Create a dropdown to select a saved file
    selected_file = st.sidebar.selectbox("Review Past Patient Records:", output_files)
    
    if selected_file:
        file_path = os.path.join(OUTPUTS_DIR, selected_file)
        
        # Read the file data for the download button
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            
        # Display logic depending on file type
        if selected_file.lower().endswith('.txt'):
            st.sidebar.markdown("### 📋 Clinical Report")
            # Decode the bytes to string for display in the UI
            st.sidebar.text(file_bytes.decode('utf-8', errors='ignore'))
            mime_type = "text/plain"
        else:
            # Show the thumbnail in the sidebar for images
            st.sidebar.image(file_path, caption=selected_file, use_container_width=True)
            mime_type = "image/jpeg"
            
        # Group the Download and Delete buttons nicely
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            # Add a universal download button
            st.download_button(
                label="Download",
                data=file_bytes,
                file_name=selected_file,
                mime=mime_type,
                use_container_width=True
            )
            
        with col2:
            # Add the delete button
            if st.button("️ Delete", type="primary", use_container_width=True):
                try:
                    os.remove(file_path)
                    # Instantly refresh the Streamlit app so the deleted file disappears from the dropdown
                    st.rerun() 
                except Exception as e:
                    st.error(f"Error deleting file: {e}")

# --- MAIN UI WORKFLOW ---
uploaded_file = st.file_uploader("Upload Retinal Scan (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Save original scan to the 'Inputs' folder
    input_path = os.path.join(INPUTS_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the image from memory for UI
    img = Image.open(uploaded_file).convert('RGB')
    
    # Create two columns side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original Patient Scan", use_container_width=True)
        
    with col2:
        st.subheader("AI Analysis Results")
        
        with st.spinner("Processing image through neural networks..."):
            # 1. Base Prediction
            p_idx, p_conf, p_tensor = predict(img, INITIAL)
            diag = Initial_translate.get(p_idx, "Unknown")
            
            # Use progress bar as a visual "Confidence Meter"
            st.metric("Primary Diagnosis", diag)
            st.progress(float(p_conf), text=f"Confidence: {p_conf*100:.2f}%")
            
            active_model = INITIAL
            final_idx = p_idx
            
            # Initialize clinical report text
            report_text = f"PATIENT SCAN FILE: {uploaded_file.name}\n"
            report_text += f"{'-'*40}\n"
            report_text += f"PRIMARY DIAGNOSIS: {diag}\n"
            report_text += f"CONFIDENCE:        {p_conf*100:.2f}%\n"
            
            # 2. Cascade Logic
            if diag == "AMD":
                st.warning("AMD Detected. Cascading to Specialist Model...")
                s_idx, s_conf, _ = predict(img, AMD)
                sub_diag = AMD_translate[s_idx]
                
                st.metric("AMD Classification", sub_diag)
                st.progress(float(s_conf), text=f"Sub-Confidence: {s_conf*100:.2f}%")
                
                active_model = AMD
                final_idx = s_idx
                
                # Append cascade results to report
                report_text += f"{'-'*40}\n"
                report_text += f"SUB-CLASSIFICATION: AMD - {sub_diag}\n"
                report_text += f"SUB-CONFIDENCE:     {s_conf*100:.2f}%\n"
                
            elif diag == "Diabetes":
                st.warning("Diabetes Detected. Cascading to Specialist Model...")
                s_idx, s_conf, _ = predict(img, DME)
                sub_diag = DME_translate[s_idx]
                
                st.metric("DME Findings", sub_diag)
                st.progress(float(s_conf), text=f"Sub-Confidence: {s_conf*100:.2f}%")
                
                active_model = DME
                final_idx = s_idx
                
                # Append cascade results to report
                report_text += f"{'-'*40}\n"
                report_text += f"SUB-CLASSIFICATION: Diabetes - {sub_diag}\n"
                report_text += f"SUB-CONFIDENCE:     {s_conf*100:.2f}%\n"

            # 3. Save the Clinical Report to Outputs folder automatically
            # Remove extension from original filename for the text file name
            base_name = os.path.splitext(uploaded_file.name)[0]
            report_filename = f"diagnosis_{base_name}.txt"
            report_path = os.path.join(OUTPUTS_DIR, report_filename)
            
            with open(report_path, "w") as f:
                f.write(report_text)
                
            st.success("Clinical Diagnosis Report automatically saved to Archive!")

        # 4. Grad-CAM Generation
        st.divider()
        if st.button("Generate Pathology Heatmap", type="primary"):
            with st.spinner("Calculating Gradient Class Activation Maps..."):
                
                # Save the generated heatmap to the 'Outputs' folder
                heatmap_filename = f"heatmap_{base_name}.png"
                output_path = os.path.join(OUTPUTS_DIR, heatmap_filename)
                
                # Pass the output path to your backend function
                GradCAMSave(active_model, p_tensor, img, final_idx, output_path)
                
                # Show image from the Outputs folder
                st.image(output_path, caption="Grad-CAM Pathology Localization", use_container_width=True)
                
                # Force the Streamlit UI to refresh so the new file instantly appears in the sidebar archive
                st.success(f"Heatmap archived! Check the sidebar to view and download it. You may have to reload your page first.")

st.markdown("---")
st.caption("*Disclaimer: This tool is an AI diagnostic assistant and should not replace a professional doctor's opinion.*")