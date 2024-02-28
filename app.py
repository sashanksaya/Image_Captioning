import streamlit as st
import requests
import numpy as np
from PIL import Image
from model import get_caption_model, generate_caption


@st.cache(allow_output_mutation=True)
def get_model():
    return get_caption_model()
caption_model = get_model()

img_url = st.text_input(label='https://images.unsplash.com/photo-1518895949257-7621c3c786d7?q=80&w=2788&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')

if (img_url != "") or (img_url != None):
    img = Image.open(requests.get(img_url, stream=True).raw)
    st.image(img)

    img = np.array(img)
    pred_caption = generate_caption(img, caption_model)
    st.write(pred_caption)
