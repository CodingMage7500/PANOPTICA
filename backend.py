import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

# ========================
# SETUP & DEVICE
# ========================
device = torch.device("cpu")

def clear_screen():
    # Check if the operating system is Windows ('nt') or Posix (Linux/macOS)
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

# ========================
# DICTIONARIES (Global Constants)
# ========================
Initial_translate = {0: "AMD", 1: "Cataracts", 2: "Diabetes", 3: "Glaucoma", 4: "Normal"}
AMD_translate = {0: "Dry", 1: "Wet"}
DME_translate = {0: "No DME", 1: "DME"}

# ========================
# THE MODEL ARCHITECTURES
# ========================
class model(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.effnet = torch.hub.load('pytorch/vision:v0.17.2', 'efficientnet_b0', weights=None)
        self.effnet.classifier = nn.Sequential(
            nn.Linear(self.effnet.classifier[1].in_features, 248),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(248, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(50, classes)
        )
    def forward(self, x):
        return self.effnet(x)

class modelX(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.effnet = torch.hub.load('pytorch/vision:v0.17.2', 'efficientnet_b0', weights='IMAGENET1K_V1')
        [param.requires_grad_(False) for param in self.effnet.features.parameters()]
        
        self.effnet.classifier = nn.Sequential(
            nn.Linear(self.effnet.classifier[1].in_features, 248),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(248, 248),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(248, 248),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(248, 100),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(50, classes)
        )
    def forward(self, x):
        return self.effnet(x)

# ========================
# MODEL LOADER
# ========================
def load_models():
    """Instantiates models and loads weights. Wrapped in a function for Streamlit caching."""
    m_initial = model(5).to(device)
    m_amd = model(2).to(device)
    m_dme = modelX(2).to(device)

    try:
        print("Booting... please stand by, loading models...")
        m_initial.load_state_dict(torch.load('MODELS/Initial.pth', map_location=device, weights_only=True))
        print("Initial model loaded...")
        m_amd.load_state_dict(torch.load('MODELS/AMD.pth', map_location=device, weights_only=True))
        print("First supporting model loaded...")
        m_dme.load_state_dict(torch.load('MODELS/DME.pth', map_location=device, weights_only=True))
        print("Second supporting model loaded...")
        print("Models loaded! Booting helper functions...")
        time.sleep(1)
    except FileNotFoundError:
        print("WARNING: Could not find model weight files. Make sure the 'MODELS' folder exists!")

    # Set to evaluation mode
    m_initial.eval()
    m_amd.eval()
    m_dme.eval()

    return m_initial, m_amd, m_dme

# ========================
# HELPER FUNCTIONS
# ========================
def predict(image, model_obj):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model_obj(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
    return pred.item(), conf.item(), img_t

def GradCAMSave(model_obj, img_tensor, pil_img, pred_idx, save_path):
    # Unfreeze layers temporarily so Grad-CAM can calculate gradients
    for param in model_obj.effnet.features.parameters():
        param.requires_grad = True
        
    target_layers = [model_obj.effnet.features[-1]]
    cam = GradCAM(model=model_obj, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    
    # Resize original image to match tensor and convert to float [0-1]
    rgb_img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

# ========================
# MAIN PIPELINE LOOP (For Terminal Use)
# ========================
def run_pipeline(INITIAL, AMD, DME):
    while True:
        print("\n" + "="*40)
        img_path = input("Enter image path to scan or 'Q' to quit: >>> ").strip()
        
        if img_path.upper() == 'Q':
            print("Exiting software. Have a great day!")
            break
            
        if not os.path.exists(f"INPUT/{img_path}"):
            print("File not found. Please check the path and try again.")
            continue
            
        try:
            # 1. Load Image
            pil_img = Image.open(f"INPUT/{img_path}").convert('RGB')
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 2. Create specific output folder
            save_dir = os.path.join("OUTPUTS", base_name)
            os.makedirs(save_dir, exist_ok=True)
            
            print("Analyzing scan...")
            
            # 3. Run INITIAL Classifier
            pred_idx, conf, img_tensor = predict(pil_img, INITIAL)
            diagnosis = Initial_translate.get(pred_idx, "Unknown")
            
            final_report = f"Base Diagnosis: {diagnosis} (Confidence: {conf*100:.2f}%)\n"
            active_model = INITIAL # Keep track of which model made the final call for GradCAM
            
            # 4. The Cascade Logic
            if diagnosis == "AMD":
                print("AMD Detected. Running sub-classifier for Wet vs Dry...")
                sub_idx, sub_conf, _ = predict(pil_img, AMD)
                sub_diag = AMD_translate.get(sub_idx, "Unknown")
                final_report += f"Sub-Diagnosis: {sub_diag} AMD (Confidence: {sub_conf*100:.2f}%)\n"
                active_model = AMD
                pred_idx = sub_idx
                
            elif diagnosis == "Diabetes":
                print(" Diabetes Detected. Running sub-classifier for DME...")
                sub_idx, sub_conf, _ = predict(pil_img, DME)
                sub_diag = DME_translate.get(sub_idx, "Unknown")
                final_report += f"Sub-Diagnosis: {sub_diag} (Confidence: {sub_conf*100:.2f}%)\n"
                active_model = DME
                pred_idx = sub_idx

            print("RESULTS:")
            print(final_report)
            
            # 5. Run Grad-CAM and Save Image
            cam_path = os.path.join(save_dir, f"{base_name}_GradCAM.jpg")
            GradCAMSave(active_model, img_tensor, pil_img, pred_idx, cam_path)
            
            # 6. Save Text Report
            txt_path = os.path.join(save_dir, f"{base_name}_report.txt")
            with open(txt_path, "w") as f:
                f.write(final_report)
                
            print(f"Saved results and heatmap to: {save_dir}/")
            print()
            print("-------------------------------")
            print("How to interpret confidence")
            print("-------------------------------")
            print(">90%: Extremely sure")
            print("80%-90%: Very sure")
            print("70%-80%: Moderately confident")
            print("70%-60%: Somewhat unsure")
            print("<60%: Entirely unsure")
            print("-------------------------------")
            print("This result should NOT be regarded as the replacement for a doctor's opinion.\n\n")
            
            input("Press ENTER to proceed.")
            clear_screen()
            
        except Exception as e:
            print(f" An error occurred processing the image: {e}")

# ========================
# EXECUTION SHIELD
# ========================
# This blocks the terminal loop from running if this file is imported by Streamlit!
if __name__ == "__main__":
    clear_screen()
    print("Booted! Software ready.")
    time.sleep(0.5)
    clear_screen()
    print("================================================")
    print("    WELCOME TO THE EYE DISEASE CLASSIFIER")
    print("      Supporting five different diseases")
    print("================================================")
    
    input("Press ENTER to initiate software...")
    clear_screen()
    
    # Load models directly into variables here, then pass them to the pipeline
    INITIAL_MODEL, AMD_MODEL, DME_MODEL = load_models()
    run_pipeline(INITIAL_MODEL, AMD_MODEL, DME_MODEL)