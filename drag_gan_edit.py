import torch
import torch.optim as optim

def drag_gan_edit(
    generator,
    mapping_network,
    initial_latent,
    source_points,
    target_points,
    num_steps=2000,
    lr=0.005,
    device="cuda",
    is_w_input=False,
    log_resolution=7
):
    if is_w_input:
        w = initial_latent.detach().clone().requires_grad_(True)
    else:
        w = mapping_network(initial_latent).detach().clone().requires_grad_(True)
    optimizer = optim.Adam([w], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        w_broadcast = [w for _ in range(log_resolution)]
        noise = []
        resolution = 4
        for i in range(log_resolution):
            n1 = torch.randn(1, 1, resolution, resolution).to(device) if i != 0 else None
            n2 = torch.randn(1, 1, resolution, resolution).to(device)
            noise.append((n1, n2))
            resolution *= 2
        img = generator(w_broadcast, noise)[0]
        img = (img * 0.5 + 0.5).clamp(0, 1)

        loss = 0
        for (x_s, y_s), (x_t, y_t) in zip(source_points, target_points):
            src_val = img[:, y_s, x_s]
            tgt_val = img[:, y_t, x_t]
            loss += torch.norm(src_val - tgt_val, p=2)
        loss.backward()
        optimizer.step()

    return img.detach().cpu(), w.detach().cpu()
