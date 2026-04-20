import streamlit as st
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms as tr
from generator import ResnetGenerator

@st.cache_resource
def load_generators():
    G_A2B = ResnetGenerator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d, n_blocks=9)
    G_B2A = ResnetGenerator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d, n_blocks=9)
    G_A2B.load_state_dict(torch.load("generator_a2b.pth", map_location=torch.device('cpu'), weights_only=False))
    G_B2A.load_state_dict(torch.load("generator_b2a.pth", map_location=torch.device('cpu'), weights_only=False))
    G_A2B.eval()
    G_B2A.eval()
    return G_A2B, G_B2A

def transform_image(image):
    transform = tr.Compose([
        tr.Resize(256),
        tr.ToTensor(),
        tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def inverse_transform(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return tr.ToPILImage()(tensor)

def process_image(img, direction):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    G_A2B, G_B2A = load_generators()
    img_tensor = transform_image(img)
    if direction == "Из яблока в апельсин":
        output = G_A2B(img_tensor)
    else:
        output = G_B2A(img_tensor)
    return inverse_transform(output)

st.set_page_config(page_title="Яблоко? Апельсин!", layout="centered")

if "gallery_offset" not in st.session_state:
    st.session_state["gallery_offset"] = 0
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
if "is_example" not in st.session_state:
    st.session_state["is_example"] = False
if "result_image" not in st.session_state:
    st.session_state["result_image"] = None

st.title("Яблоко? Апельсин!")
st.caption("Загрузите фото яблока или апельсина — нейросеть превратит одно в другое")

direction = st.radio(
    "Направление преобразования:",
    ["Из яблока в апельсин", "Из апельсина в яблоко"],
    horizontal=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Исходное изображение")

    if st.session_state.get("selected_image") is not None:
        if st.session_state["is_example"]:
            img = Image.open(st.session_state["selected_image"])
        else:
            img = Image.open(st.session_state["selected_image"])
        st.image(img, width='stretch')

    uploaded_file = st.file_uploader("Загрузите своё фото", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
            st.session_state["last_uploaded"] = uploaded_file.name
            st.session_state["selected_image"] = uploaded_file
            st.session_state["is_example"] = False
            img = Image.open(uploaded_file).convert('RGB')
            st.session_state["result_image"] = process_image(img, direction)
            st.rerun()

    st.markdown("---")
    st.caption("Или выберите из примеров:")

    if direction == "Из яблока в апельсин":
        examples_dir = "galery/Apples"
    else:
        examples_dir = "galery/Oranges"

    if os.path.exists(examples_dir):
        all_images = [f for f in os.listdir(examples_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        start_idx = st.session_state["gallery_offset"]
        end_idx = min(start_idx + 3, len(all_images))
        current_images = all_images[start_idx:end_idx]

        cols = st.columns(3)
        for i, img_name in enumerate(current_images):
            img_path = os.path.join(examples_dir, img_name)
            img = Image.open(img_path)
            with cols[i]:
                st.image(img, width='stretch')
                if st.button("Выбрать", key=f"select_{img_name}"):
                    if st.session_state.get("last_selected") != img_path:
                        st.session_state["last_selected"] = img_path
                        st.session_state["selected_image"] = img_path
                        st.session_state["is_example"] = True
                        img = Image.open(img_path).convert('RGB')
                        st.session_state["result_image"] = process_image(img, direction)
                        st.rerun()

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("Заменить примеры"):
                st.session_state["gallery_offset"] = (start_idx + 3) % len(all_images)
                st.rerun()

with col2:
    st.subheader("Результат")
    if st.session_state.get("result_image") is not None:
        st.image(st.session_state["result_image"], width='stretch')