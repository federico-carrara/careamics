import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from careamics.losses.lvae.losses import get_reconstruction_loss, _reconstruction_loss_musplit_denoisplit
from careamics.lightning import VAEModule
    

def get_dset_predictions(
    model: VAEModule,
    dloader: DataLoader,
    mmse_count: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Get patch-wise predictions from a model for the entire dataset.

    Parameters
    ----------
    model : VAEModule
        Lightning model used for prediction.
    dloader : DataLoader
        DataLoader to predict on.
    mmse_count : int, optional
        Number of samples to generate for each input and then to average over for
        MMSE estimation, by default 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing:
            - predictions: Predicted images for the dataset.
            - predictions_std: Standard deviation of the predicted images.
    """
    predictions = []
    predictions_std = []
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting patches"):
            inp, tar = batch
            inp = inp.cuda()
            if tar:
                tar = tar.cuda()

            rec_img_list = []
            logvar_img_list = []
            for _ in range(mmse_count):
                # get model output
                rec, out, _ = model(inp)

                # get reconstructed img
                if model.model.predict_logvar is None:
                    rec_img = out
                    logvar = torch.tensor([-1])
                else:
                    rec_img, logvar = torch.chunk(out, chunks=2, dim=1)
                rec_img_list.append(rec_img.cpu().unsqueeze(0)) 
                logvar_img_list.append(logvar.cpu().unsqueeze(0))

            # aggregate results
            samples = torch.cat(rec_img_list, dim=0)
            mmse_imgs = torch.mean(samples, dim=0)
            mmse_std = torch.std(samples, dim=0)
            predictions.append(mmse_imgs.cpu().numpy())
            predictions_std.append(mmse_std.cpu().numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(predictions_std, axis=0),
    )
