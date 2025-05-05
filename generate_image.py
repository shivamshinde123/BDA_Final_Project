    
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from StyleGAN2_Implementation import Generator
from StyleGAN2_Implementation import MappingNetwork

if __name__ == "__main__":


    Z_DIM = 256
    W_DIM = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOG_RESOLUTION = 7

    gen  = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
    gen.load_state_dict(torch.load(os.path.join('stylegan2_checkpoints', 'gen.pth'), map_location=DEVICE))

    mapping_network  = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
    mapping_network.load_state_dict(torch.load(os.path.join('stylegan2_checkpoints', 'mapping_network.pth'), map_location=DEVICE))

    gen.eval()
    mapping_network.eval()


    for k in range(10):

        # --- 1. Sample latent vector z ---
        z = torch.randn(1, Z_DIM).to(DEVICE)

        # --- 2. Map z to w using mapping_network ---
        with torch.no_grad():
            w = mapping_network(z)  # shape: (1, W_DIM)

        # --- 3. Broadcast w for each resolution block ---
        w_broadcast = [w for _ in range(LOG_RESOLUTION)]

        # --- 4. Prepare noise for each block ---
        input_noise = []
        for i in range(LOG_RESOLUTION):
            res = 4 * 2 ** i
            noises = [torch.randn(1, 1, res, res).to(DEVICE) for _ in range(2)]
            input_noise.append(noises)

        # --- 5. Generate image ---
        with torch.no_grad():
            fake_img = gen(w_broadcast, input_noise)  # Output shape: (1, 3, H, W)

        # --- 6. Denormalize and save/display image ---
        img = (fake_img + 1) / 2  # Convert from [-1, 1] to [0, 1]
        os.makedirs("Generated_Images",exist_ok=True)
        save_image(img, os.path.join("Generated_Images", f"generated_image_{k}.png"))
        print(f"Image saved as generated_image_{k}.png")
        plt.imshow(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
