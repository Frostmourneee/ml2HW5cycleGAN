import streamlit as st
import torch, os
import torch.nn as nn
from PIL import Image
import numpy as np
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

st.set_page_config(page_title="Яблоко? Апельсин!", layout="centered")
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
    st.session_state["is_example"] = False

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
    uploaded_file = st.file_uploader("Загрузите своё фото", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.caption("Или выберите из примеров:")

    if "Из яблока" in direction:
        examples_dir = "galery/Apples"
    else:
        examples_dir = "galery/Oranges"

    example_images = [f for f in os.listdir(examples_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    # image_items = []
    # for img_file in example_images:
    #     img_path = os.path.join(examples_dir, img_file)
    #     image_items.append({
    #         "src": f"data:image/jpeg;base64,{get_image_base64(img_path)}",
    #         "title": img_file
    #     })
    #
    # clicked_index = streamlit_image_gallery(
    #     images=image_items,
    #     max_cols=3,
    #     gap=10,
    #     key="example_gallery"
    # )

    # if clicked_index is not None:
    #     example_path = os.path.join(examples_dir, example_images[clicked_index])
    #     example_img = Image.open(example_path)
    #     st.image(example_img, caption=example_images[clicked_index], use_container_width=True)
    #     st.session_state["selected_image"] = example_img
    #     st.session_state["is_example"] = True

with col2:
    st.subheader("Результат")
    st.caption(" ")

if st.button("Преобразовать", type="primary"):
    st.info("Модель загружена. Нажмите кнопку для преобразования.")
else:
    st.info("Интерфейс-заглушка. Модель пока не подключена.")