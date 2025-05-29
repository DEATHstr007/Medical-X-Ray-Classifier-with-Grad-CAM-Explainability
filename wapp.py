import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Define device (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
from bmodel import XRayClassifier
model = XRayClassifier().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Define preprocessing
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Streamlit app
st.title("Medical X-ray Classifier")
uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)
    
    # Preprocess and predict
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
    
    # Display prediction
    class_names = ['Normal', 'Pneumonia']
    pred = output.argmax(dim=1).item()
    st.write(f"**Prediction:** {class_names[pred]}")
