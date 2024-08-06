import torch
from torchvision import transforms
import wandb
from diffusers import DDIMScheduler
from tqdm import tqdm
import copy
from condition_dataset import ConditionalFramesDataset
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
dataset = ConditionalFramesDataset("frames_dataset_skip", 3)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = UNet().to(device)
master_params = list(model.parameters())
optimizer = torch.optim.AdamW(master_params, lr=1e-4, weight_decay=0.05)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


ema_rate = [0.9999]
ema_params = [copy.deepcopy(master_params) for _ in range(len(ema_rate))]

# TODO: Enforce 0 terminal SNR
beta_min = 0.0001
beta_max = 0.02
diffusion_steps = 500
betas = torch.linspace(beta_min, beta_max, diffusion_steps).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def forward(x0, t, eta):
    n, c, h, w = x0.size()
    a_bar = alpha_bars[t]
    noisy = (
        torch.sqrt(a_bar).reshape(n, 1, 1, 1) * x0
        + torch.sqrt(1 - a_bar).reshape(n, 1, 1, 1) * eta
    )
    return noisy


wandb.init(project="multimodal", mode="online")

model.train()
# Training loop
for epochs in range(35):
    for prev_image, action_window, target_image in dataloader:
        optimizer.zero_grad()

        # Noisy image
        x = target_image.to(device)
        n = x.size(0)
        eta = torch.randn_like(x, device=device)
        t = torch.randint(0, diffusion_steps, (n,), device=device)
        x = forward(x, t, eta)

        # Conditioning signal
        mask = torch.bernoulli(torch.full((x.size(0),), 0.9)).to(
            device
        )  # 10% unconditioned
        image_mask = mask.view(-1, 1, 1, 1)
        ic = image_mask * prev_image.to(device)
        x = torch.cat((x, ic), dim=1)
        text_mask = mask.view(-1, 1, 1)  # [batch, sequence, vocab]
        tc = text_mask * action_window.to(device)

        # TODO: v-prediction objective
        eta_pred = model(x, t, tc)
        loss = torch.nn.functional.mse_loss(eta_pred, eta)
        wandb.log({"MSE loss": loss.item()})
        loss.backward()
        optimizer.step()
        for rate, params in zip(ema_rate, ema_params):
            update_ema(params, master_params, rate=rate)

wandb.finish()
torch.save(model.state_dict(), "unet_model.pth")

# model.load_state_dict(torch.load('unet_model.pth'))
model.eval()

# DDIM sampling
x = torch.randn(batch_size, 3, 120, 160, device=device)
w = 2.0

prev_frame, actions, target_frame = next(iter(dataloader))
prev_frame = prev_frame.repeat(2, 1, 1, 1).to(device)
actions = actions.repeat(2, 1, 1).to(device)

mask = torch.ones((actions.size(0),), device=device)
mask[batch_size:] = 0
image_mask = mask.view(-1, 1, 1, 1)
text_mask = mask.view(-1, 1, 1)  # [batch, sequence, vocab]

scheduler = DDIMScheduler(num_train_timesteps=diffusion_steps, clip_sample=True, set_alpha_to_one=True)
scheduler.set_timesteps(50)


for t in tqdm(scheduler.timesteps):
    x = x.repeat(2, 1, 1, 1)

    with torch.no_grad():
        noise_pred = model(
            torch.cat((x, image_mask * prev_frame), dim=1),
            t.repeat(actions.size(0)).to(device),
            text_mask * actions,
        )

    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
    noise_pred = noise_pred_text + w * (noise_pred_text - noise_pred_uncond)

    x = scheduler.step(noise_pred, t, x[:batch_size]).prev_sample


invTrans = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
        transforms.ToPILImage(),
    ]
)

pil_images = list(map(lambda x: invTrans(x), x))
for idx, i in enumerate(pil_images):
    i.save(f"generations/{idx}.png")

cond_images = list(map(lambda x: invTrans(x), prev_frame[:batch_size]))
for idx, i in enumerate(cond_images):
    i.save(f"conditions/{idx}.png")

cond_images = list(map(lambda x: invTrans(x), target_frame))
for idx, i in enumerate(cond_images):
    i.save(f"targets/{idx}.png")

torch.save(actions[:batch_size], "conditions/actions.pt")
