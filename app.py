import os
import torch
import numpy as np
from PIL import Image
import streamlit as st
from drag_gan_edit import drag_gan_edit
from streamlit_drawable_canvas import st_canvas
from StyleGAN2_Implementation import Generator, MappingNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 256  
W_DIM = 256
LOG_RESOLUTION = 7  

# --- Load models ---
gen  = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
gen.load_state_dict(torch.load(os.path.join('stylegan2_checkpoints', 'gen.pth'), map_location=DEVICE))

mapping_network  = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
mapping_network.load_state_dict(torch.load(os.path.join('stylegan2_checkpoints', 'mapping_network.pth'), map_location=DEVICE))

gen.eval()
mapping_network.eval()

def generate_image(z, gen, mapping_network):
    with torch.no_grad():
        w = mapping_network(z)
        w_broadcast = [w for _ in range(LOG_RESOLUTION)]
        noise = []
        resolution = 4
        for i in range(LOG_RESOLUTION):
            n1 = torch.randn(1, 1, resolution, resolution).to(DEVICE) if i != 0 else None
            n2 = torch.randn(1, 1, resolution, resolution).to(DEVICE)
            noise.append((n1, n2))
            resolution *= 2
        img = gen(w_broadcast, noise)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return img

st.title("DragGAN Interactive Editing (Multi-Iteration)")

if "z" not in st.session_state:
    st.session_state.z = torch.randn(1, Z_DIM).to(DEVICE)
if "w" not in st.session_state:
    st.session_state.w = None
if "edit_history" not in st.session_state:
    st.session_state.edit_history = []
if "last_img" not in st.session_state:
    st.session_state.last_img = None

if st.button("Generate New Image"):
    st.session_state.z = torch.randn(1, Z_DIM).to(DEVICE)
    st.session_state.w = None
    st.session_state.edit_history = []
    st.session_state.last_img = None

# Generate current image
if st.session_state.w is None or st.session_state.last_img is None:
    img = generate_image(st.session_state.z, gen, mapping_network)
    st.session_state.last_img = img
else:
    img = st.session_state.last_img

st.image(img, caption="Current Image", use_container_width=True)

# Draw all previous points for user feedback
if len(st.session_state.edit_history) > 0:
    st.write("Previous iterations:")
    for i, (src, tgt) in enumerate(st.session_state.edit_history):
        st.write(f"Iteration {i+1}: Source {src} → Target {tgt}")

# Select points for this iteration
st.write("Draw a source point (circle) and a target point (circle) for this iteration:")
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=10,
    stroke_color="#e00",
    background_image=Image.fromarray(img),
    update_streamlit=True,
    height=img.shape[0],
    width=img.shape[1],
    drawing_mode="circle",
    key="canvas",
)

if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    if len(objects) >= 2:
        source = (int(objects[0]["left"]), int(objects[0]["top"]))
        target = (int(objects[1]["left"]), int(objects[1]["top"]))
        st.write(f"Source: {source}, Target: {target}")

        if st.button("Add Drag Iteration"):
            st.session_state.edit_history.append((source, target))
            st.success("Iteration added. You can add more or proceed to editing.")

if st.button("Run DragGAN Edit (Apply All Iterations)"):
    if len(st.session_state.edit_history) == 0:
        st.warning("Please add at least one drag iteration.")
    else:
        # Start from z, then use w for subsequent edits
        current_latent = st.session_state.z
        is_w_input = False
        last_img = None
        for idx, (source, target) in enumerate(st.session_state.edit_history):
            img_tensor, w = drag_gan_edit(
                gen, mapping_network, current_latent, [source], [target],
                device=DEVICE, is_w_input=is_w_input, log_resolution=LOG_RESOLUTION
            )
            last_img = img_tensor.permute(1, 2, 0).numpy()
            last_img = (last_img * 255).astype(np.uint8)
            current_latent = w
            is_w_input = True  # After the first, always use w
        st.session_state.last_img = last_img
        st.session_state.w = current_latent
        st.image(last_img, caption="Final Edited Image", use_container_width=True)
        st.success("Editing complete!")

if st.button("Reset All"):
    st.session_state.edit_history = []
    st.session_state.w = None
    st.session_state.last_img = None
    st.experimental_rerun()


