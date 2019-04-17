import argparse
import os

import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from torch import optim
from torchvision.utils import save_image
from Problem3.vae.ignite_engines import create_vae_trainer
from Problem3.vae.svhn import get_data_loader
from Problem3.vae.vae import VAE, loss_fn


LOG_EVERY = 200


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(device, z_dim=args.z_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    train_loader, valid_loader, _ = get_data_loader("svhn", args.batch_size)
    trainer = create_vae_trainer(model, optimizer, loss_fn, device)
    check_pointer = ModelCheckpoint(dirname="checkpoints",
                                    filename_prefix='vae',
                                    save_interval=5,
                                    create_dir=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def generate_image(engine):
        with torch.no_grad():
            z = torch.randn(size=(arguments.batch_size, args.z_dim), device=device)
            generated_sample = model.decode(z)
            image_name = "generated_{}.png".format(str(engine.state.epoch))
            save_image(generated_sample, os.path.join(arguments.gen_images_dir, image_name))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_train_loss(engine):
        if engine.state.iteration % LOG_EVERY == 0:
            loss, mse, kld = engine.state.output
            print("Epoch: {}, Iteration: {}, Loss: {}, MSE: {}, KLD: {}".format(
                engine.state.epoch, engine.state.iteration, loss, mse, kld
            ))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, check_pointer, {"mymodel": model})
    trainer.run(train_loader, max_epochs=args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gen_images_dir", type=str, default=".")
    arguments = parser.parse_args()
    main(arguments)
