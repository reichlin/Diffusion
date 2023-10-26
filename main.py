import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from PIL import Image

from torchvision import datasets as torch_datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from datasets import Dataset_Toy
from diffusion_model import DDPM
from utils import create_dirs
import os
import ipdb




experiments = {0: '2d_swiss_roll', 1: '3d_swiss_roll', 2: 'two_moons', 3: 'cifar10'}

parser = argparse.ArgumentParser()

parser.add_argument('--exp', default='swiss_roll', type=str)
parser.add_argument('--T', default=1000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--beta_min', default=0.0001, type=float)
parser.add_argument('--beta_max', default=0.001, type=float) #0.02

parser.add_argument('--save-every', default=10, type=int) #0.02
parser.add_argument('--checkpoints-dir', default='checkpoints', type=str) #0.02

args = parser.parse_args()

#beta_min, beta_max = 1e-4, 0.02
beta_min, beta_max = args.beta_min, args.beta_max
T = args.T #1000
batch_size = args.batch_size #128
EPOCHS = 1000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

exp = args.exp
if exp in ['swiss-roll']:
    n_samples, noise = 100000, 0.0
    input_dim = [3] if exp == '3d-swiss-roll' else [2]
    dataset = Dataset_Toy(exp, n_samples, noise)
    cnn = False
elif exp in ['cifar']:
    dataset = torch_datasets.CIFAR10(".", transform=transforms.ToTensor(), download=True)
    input_dim = [3, 32, 32]

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#exp_name = "T="+str(T)+"_BS="+str(batch_size)+"_beta_min="+str(beta_min)+"_beta_max="+str(beta_max)
exp_name = f"{args.exp}_T={T}_beta_min={args.beta_min}_beta_max={args.beta_max}"
print(f"Training: {exp_name}")

CHECKPOINTS_DIR = args.checkpoints_dir
MODEL_DIR = os.path.join(CHECKPOINTS_DIR, exp_name)
FIGURES_DIR = os.path.join(MODEL_DIR, 'figures')

create_dirs(CHECKPOINTS_DIR)
create_dirs(MODEL_DIR)
create_dirs(FIGURES_DIR)

MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')

writer = SummaryWriter(f"./logs/{exp_name}")

model = DDPM(input_dim, T, beta_min, beta_max, device).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)


if exp in ['cifar']:
    list_imgs = []
    for batch in dataloader:
        list_imgs.append(batch[0].numpy())
    all_imgs = np.concatenate(list_imgs, 0)
    avg_pixels = torch.from_numpy(np.mean(all_imgs, (0, 2, 3))).float().to(device).view(1, -1, 1, 1)
    std_pixels = torch.from_numpy(np.std(all_imgs, (0, 2, 3))).float().to(device).view(1, -1, 1, 1)


logs_idx = 0
for epoch in tqdm(range(1, EPOCHS + 1)):
    tot_loss = 0
    for batch in dataloader:
        if exp in ['cifar']:
            x0 = batch[0].to(device) if len(batch) > 1 else batch.to(device)
            x0 = (x0 - avg_pixels) / std_pixels
        else:
            x0 = batch.to(device)

        ipdb.set_trace()
        t = torch.randint(1, T+1, size=(x0.shape[0], 1)).to(device)
        epsilon = torch.randn(size=x0.shape).to(device)

        loss = model.get_loss(x0, t, epsilon)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tot_loss += loss.detach().cpu().item()

    writer.add_scalar("loss", tot_loss/len(dataloader), logs_idx)
    logs_idx += 1

    if epoch % args.save_every == 0:
        torch.save(model, MODEL_PATH)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.3f}")

    if epoch % 10 == 0:
        x_time = model.sample(10000, device)
        if exp in ['swiss-roll', '3d-swiss-roll', 'moons']:
            x_real = dataset.X.detach().cpu().numpy()
            fig, ax = plt.subplots()
            ax.scatter(x_real[:, 0], x_real[:, 1])
            ax.scatter(x_time[-1][:, 0], x_time[-1][:, 1])
            writer.add_figure("generated_data", fig, epoch)
            fig.savefig(os.path.join(FIGURES_DIR, f'{epoch}.png'))
            plt.close(fig=fig)

            #imgs = []
            if True:
                for t in range(len(x_time)):
                    save_dir = os.path.join(FIGURES_DIR, f'epoch={epoch}')
                    create_dirs(save_dir)

                    fig, ax = plt.subplots()
                    if x_time[t].shape[-1] == 2:
                        ax.scatter(x_time[t][:, 0], x_time[t][:, 1])
                    elif x_time[t].shape[-1] == 3:
                        ax = fig.add_subplot(projection='3d')
                        ax.view_init(elev=10, azim=90)
                        ax.scatter(x_time[t][:, 0], x_time[t][:, 1], x_time[t][:, 2])
                    fig.savefig(os.path.join(save_dir, f"{epoch}_t={T-t}.png"))
                    #plt.savefig("gifs_" + save_folder_name + "/t=" + str(T - t) + ".png")
                    #imgs.append(Image.open("gifs_" + save_folder_name + "/t=" + str(T - t) + ".png"))
                    plt.close(fig=fig)

            #imgs[0].save("gifs_" + save_folder_name + "/process.gif", save_all=True, append_images=imgs, optimize=False, duration=100, loop=0)
            plt.close('all')
        else:
            gen_imgs = x_time[-1] * std_pixels.detach().cpu().numpy() + avg_pixels.detach().cpu().numpy()
            writer.add_images("generated_data", gen_imgs[:5], epoch)

writer.close()


# ax = plt.figure().add_subplot(projection='3d', elev=7, azim=-80)
# ax.scatter(X[:,0], X[:,1], X[:,2], s=1)
# plt.show()


'''

T = 1000
beta = 0.0001, 0.02
group normalization
dropout 0.1
data augmentation with image flipping

'''