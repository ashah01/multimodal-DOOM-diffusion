import copy

import torch
import wandb
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

from tokenizer.magvit2_pytorch import VideoTokenizer
from condition_dataset import ConditionalVideoDataset
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
dataset = ConditionalVideoDataset("frames_dataset")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

tokenizer = VideoTokenizer( # use checkpoint config
    image_size=128,
    init_dim=22,
    num_res_blocks=1,
    ch_mult=(1, 2),
    z_channels=256,
    perceptual_loss_weight=0,
    use_gan=False,
    adversarial_loss_weight=0,
).to(device)

tokenizer.load("tokenizer/checkpoints/checkpoint.pt")
tokenizer.eval()

EPOCHS = 30

model = UNet().to(device)
master_params = list(model.parameters())
optimizer = torch.optim.AdamW(master_params, lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(dataloader), 5e-5)

def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


ema_rate = [0.9999]
ema_params = [copy.deepcopy(master_params) for _ in range(len(ema_rate))]

# TODO: Enforce 0 terminal SNR
beta_min = 0.0001
beta_max = 0.02
diffusion_steps = 1000
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
for epoch in range(EPOCHS):
    for video, action_window in dataloader:
        optimizer.zero_grad()

        # Noisy image
        x = video.to(device)
        n = x.size(0)

        x = tokenizer(x)
        eta = torch.randn_like(x, device=device)
        t = torch.randint(0, diffusion_steps, (n,), device=device)
        x = forward(x, t, eta)

        # Frame conditioning signal
        mask = torch.bernoulli(torch.full((x.size(0),), 0.9)).to(
            device
        )  # 10% unconditioned
        image_mask = mask.view(-1, 1, 1, 1)
        ic = image_mask * prev_image.to(device)
        x = torch.cat((x, ic), dim=1)

        # Text conditioning signal
        text_mask = mask.view(-1, 1, 1)  # [batch, sequence, vocab]
        tc = text_mask * action_window.to(device)

        # TODO: v-prediction objective
        eta_pred = model(x, t, tc)
        loss = torch.nn.functional.mse_loss(eta_pred, eta)
        wandb.log({"MSE loss": loss.item()})
        loss.backward()
        optimizer.step()
        scheduler.step()
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

scheduler = DDIMScheduler(
    num_train_timesteps=diffusion_steps, clip_sample=True, set_alpha_to_one=True
)
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
