import torch
import torch.optim as optim
# from dataloaders import get_mnist_dataloaders, get_lsun_dataloader
from .models import Generator, Discriminator
from .training import Trainer
from torchvision.utils import save_image
import os

def train(data_loader, checkpoint_folder):
    # data_loader, _ = get_mnist_dataloaders(batch_size=64)
    img_size = (32, 32, 3)

    generator = Generator(img_size=img_size, latent_dim=100, dim=32)
    discriminator = Discriminator(img_size=img_size, dim=32)

    print(generator)
    print(discriminator)

    # Initialize optimizers
    lr = 1e-4
    betas = (.5, .999)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Train model
    epochs = 300
    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
    trainer.train(data_loader, epochs, save_training_gif=True)

    # Save models
    torch.save(trainer.G.state_dict(), os.path.join(checkpoint_folder, f'G_checkpoint_{epochs}.pt'))
    torch.save(trainer.D.state_dict(), os.path.join(checkpoint_folder, f'D_checkpoint_{epochs}.pt'))


def generate_images(num_samples, checkpoint_folder, generated_images_folder):
    # Load the last model checkpoint in the checkpoint folder
    checkpoints = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pt')]
    checkpoint = os.path.join(checkpoint_folder, sorted(checkpoints)[-1])
    
    generator = Generator(img_size=(32, 32, 3), latent_dim=100, dim=16)
    generator.load_state_dict(torch.load(checkpoint))

    # Generate images and save the result
    generator.eval()
    os.makedirs(generated_images_folder, exist_ok=True)
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, 100)
            sample = generator(z)
            save_image(sample.view(3, 32, 32), os.path.join(generated_images_folder, f"sample_{i}.png"))

    print(f"Generated {num_samples} images at epoch {checkpoint.split('_')[-1].split('.')[0]}")
