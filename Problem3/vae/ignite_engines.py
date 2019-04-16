import torch
from ignite.engine import Engine
from ignite.utils import convert_tensor


def create_vae_trainer(model, optimizer, loss_fn):
    def _update(engine, data):
        model.train()
        optimizer.zero_grad()
        data = convert_tensor(data[0])
        reconstructed_data, mu, log_var = model(data)
        loss, mse, kld = loss_fn(reconstructed_data, data, mu, log_var)
        loss.backward()
        optimizer.step()
        return loss.item() / len(data), mse.item() / len(data), kld.item() / len(data)

    return Engine(_update)


def create_vae_evaluator(model, metrics):
    def _inference(engine, data):
        model.eval()
        with torch.no_grad():
            data = convert_tensor(data[0])
            reconstructed_data, mu, log_var = model(data)
            return reconstructed_data, data, mu, log_var   # This is fed to the loss function

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
