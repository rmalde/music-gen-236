import torch

from params import *

from generator import WaveGANGenerator
from discriminator import WaveGANDiscriminator


def load_gen_and_disc(path):
    ''' Load Generator and Discriminator from a single .pt file'''
    save_dict = torch.load(path)
    generator = WaveGANGenerator()
    discriminator = WaveGANDiscriminator()

    generator.load_state_dict(save_dict["gen"])
    discriminator.load_state_dict(save_dict["disc"])
    return generator, discriminator

def sample_noise(batch_size, noise_dim):
    '''Sample noise. batch_size: int, noise_dim: int'''
    z = torch.FloatTensor(batch_size, noise_dim)
    z.data.normal_()  # generating latent space based on normal distribution
    return z

def toggle_grads(model, flag):
    '''Set the requires_grad attribute in all parameters in model
        equal to flag (boolean)
    '''
    for p in model.parameters():
        p.requires_grad = flag

def get_device():
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    print(f'Using {device} device')
    return device

def save_gen_and_disc(generator, discriminator, path):
    ''' Save generator and discriminator to a single .pt file'''
    save_dict = {
        "gen": generator.state_dict(),
        "disc": discriminator.state_dict()
    }
    torch.save(
        save_dict,
        path,
    )
    print("Saved model at", path)
