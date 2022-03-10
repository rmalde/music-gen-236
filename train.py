import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad, Variable
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm
import argparse
# from warmup_scheduler import GradualWarmupScheduler

from params import *
from utils import *
from dataset import SongDataset
from generator import TransGANGenerator, WaveGANGenerator
from discriminator import TransGANDiscriminator, WaveGANDiscriminator

class Trainer():
    '''Trainer for all models. Parameters specified through command line args'''
    def __init__(
        self, dataloader, model_type=WAVEGAN, device="cpu", epochs=EPOCHS, noise_dim=NOISE_DIM,
        output_dir=MODEL_OUTPUT_DIR, output_prefix="WaveGAN", epochs_per_save=EPOCHS_PER_SAVE,
        n_critic=N_CRITIC, phase_shuffle=False, spectral_norm=False, warmup=False, style_gan=False
    ):
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.noise_dim = noise_dim
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.n_critic = n_critic
        self.epochs_per_save = epochs_per_save

        if model_type == TRANSGAN:
            self.discriminator = TransGANDiscriminator().to(self.device)
            self.generator = TransGANGenerator().to(self.device)
        else:
            shift_factor = 2 if phase_shuffle else 0
            self.discriminator = WaveGANDiscriminator(shift_factor=shift_factor, spectral_norm=spectral_norm).to(self.device)
            self.generator = WaveGANGenerator(style_gan=style_gan).to(self.device)


        self.discriminator.apply(self.init_weights)
        self.generator.apply(self.init_weights)

        
        # Setup Adam optimizers for both G and D
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=LR_G, betas=(BETA1, BETA2)
        )  
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2)
        )
        # if warmup:
        #     self.scheduler = GradualWarmupScheduler(optim, multiplier=1, total_epoch=1)

        self.n_samples_per_batch = len(dataloader)

        self.criterion = nn.BCEWithLogitsLoss()


    
    def init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            layer.weight.data.normal_(0.0, 0.02)
            if layer.bias is not None:
                layer.bias.data.fill_(0)
            layer.bias.data.fill_(0)
        elif isinstance(layer, nn.Linear):
            layer.bias.data.fill_(0)
    
    def calc_disc_loss(self, real, generated, batch_size):
        """
        Calculate Wasserstein Loss with Gradient penalty
        """
        disc_out_gen = self.discriminator(generated)
        disc_out_real = self.discriminator(real)

        #interpolate real and fakes for gradient
        epsilon = torch.rand(batch_size, 1, 1, device=device, requires_grad=True)
        mixed = epsilon * real + (1 - epsilon) * generated
        mixed_scores = self.discriminator(mixed)

        # calculate gradient penalty
        gradients = grad(
            inputs=mixed,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]

        grad_penalty = (
            PENALTY_COEFF
            * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        )
        assert not (torch.isnan(grad_penalty))
        assert not (torch.isnan(disc_out_gen.mean()))
        assert not (torch.isnan(disc_out_real.mean()))

        #w loss with grad penalty
        loss = (disc_out_gen - disc_out_real + grad_penalty).mean()
        return loss

    def calc_gen_loss(self, generated):
        """
        Wasserstein Loss for Generator
        """
        disc_output_gen = self.discriminator(generated)
        loss = -torch.mean(disc_output_gen)
        return loss
    
    def calc_disc_loss_simple(self, real, generated):
        """
        Calcualtes Discriminator loss using BCE + sigmoid.
        Faster to train than using W-Loss, but can lead to mode collapse
        """
        disc_out_gen = self.discriminator(generated)
        disc_out_real = self.discriminator(real)

        disc_loss_gen = self.criterion(disc_out_gen, torch.zeros_like(disc_out_gen))
        disc_loss_real = self.criterion(disc_out_real, torch.ones_like(disc_out_real))
        disc_loss = torch.mean(torch.stack([disc_loss_gen, disc_loss_real]))
        return disc_loss

    def calc_gen_loss_simple(self, generated):
        """
        Calcualtes Generator loss using BCE + sigmoid.
        Faster to train than using W-Loss, but can lead to mode collapse
        """
        disc_output_gen = self.discriminator(generated)
        gen_loss = self.criterion(disc_output_gen, torch.ones_like(disc_output_gen))
        return gen_loss

    def apply_zero_grad(self):
        self.generator.zero_grad()
        self.optimizer_g.zero_grad()

        self.discriminator.zero_grad()
        self.optimizer_d.zero_grad()

    
    def train(self):

        self.generator.train()
        self.discriminator.train()

        tb = SummaryWriter() #tensorboard

        for epoch in range(self.epochs):
            print('Training epoch:', epoch)
            pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for step, real in pbar:
                if real.shape[2] != SLICE_LEN: continue   #skips malformed data
                real = real.to(self.device)

                mean_disc_loss = 0

                ### DISCRIMINATOR LEARNING 

                for _ in range(self.n_critic):
                    self.apply_zero_grad()

                    toggle_grads(self.generator, False)
                    toggle_grads(self.discriminator, True)

                    batch_size = real.shape[0]
                    noise = sample_noise(batch_size, self.noise_dim).to(self.device)

                    generated = self.generator(noise)

                    disc_loss = self.calc_disc_loss(real.detach(), generated.detach(), batch_size)
                    # disc_loss = self.calc_disc_loss_simple(real, generated.detach())
                    mean_disc_loss += disc_loss.item() / self.n_critic

                    disc_loss.backward(retain_graph=True)
                    
                    self.optimizer_d.step()

                ## GENERATOR LEARNING

                self.apply_zero_grad()

                toggle_grads(self.generator, True)
                toggle_grads(self.discriminator, False)

                noise = sample_noise(batch_size, self.noise_dim).to(self.device)
                generated = self.generator(noise)

                gen_loss = self.calc_gen_loss(generated)
                gen_loss.backward()

                self.optimizer_g.step()

                toggle_grads(self.generator, False)
                toggle_grads(self.discriminator, False)
                
                # Write to tqdm and tensorboard
                pbar.set_postfix(gen_loss=gen_loss.item(), disc_loss=mean_disc_loss)
                tb.add_scalar("Gen loss", gen_loss, step*(epoch + 1))
                tb.add_scalar("Disc loss", disc_loss, step*(epoch + 1))

                # scheduler.step()
            # save model to file
            if epoch % self.epochs_per_save == 0:
                path = os.path.join(self.output_dir, f"{self.output_prefix}-{epoch}.pt")
                save_gen_and_disc(self.generator, self.discriminator, path)

        tb.close()

    
    

if __name__ == '__main__':
    '''
    Usage: train.py [-h] [--model WaveGAN] [--epochs 20]
                [--epochs_per_save 10] [--batch_size 4]
                [--n_critic 5] [--phase_shuffle] [--spectral_norm]
                [--warmup] [--style_gan]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model architecture [WaveGAN, TransGAN]. Default: WaveGAN", default=WAVEGAN)
    parser.add_argument("--epochs", help=f"Training epochs. Default: {EPOCHS}", default=EPOCHS, type=int)
    parser.add_argument("--epochs_per_save", help=f"Save model every n epochs. Default: {EPOCHS_PER_SAVE}", type=int, default=EPOCHS_PER_SAVE)
    parser.add_argument("--batch_size", help=f"Batch size. Default: {BATCH_SIZE}", default=BATCH_SIZE, type=int)
    parser.add_argument("--n_critic", help=f"n disc updates for 1 gen update. Default: {N_CRITIC}", type=int, default=N_CRITIC)
    parser.add_argument("--phase_shuffle", help="Use phase shuffle. Default: False", action="store_true")
    parser.add_argument("--spectral_norm", help="Use spectral norm. Default: False", action="store_true")
    parser.add_argument("--warmup", help="Use warmup. Default: False", action="store_true")
    parser.add_argument("--style_gan", help="Use AdaIN from StyleGAN", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.model.lower() == TRANSGAN.lower():
        model_type = TRANSGAN
    else:
        model_type = WAVEGAN


    device = get_device()
    dataset = SongDataset(load_path=os.path.join(PREPROCESSED_DATA_DIR, "train.pt"))
    dataloader = DataLoader(dataset, batch_size=args.batch_size)


    trainer = Trainer(
        dataloader=dataloader,
        model_type=model_type,
        device=device,
        epochs=args.epochs,
        epochs_per_save=args.epochs_per_save,
        n_critic=args.n_critic,
        phase_shuffle=args.phase_shuffle,
        spectral_norm=args.spectral_norm,
        warmup=args.warmup,
        style_gan=args.style_gan
    )
    trainer.train()