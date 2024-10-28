import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from careamics.lightning import VAEModule
from careamics.config.tile_information import TileInformation
    

def get_tiled_predictions(
    model: VAEModule,
    dloader: DataLoader,
    mmse_count: int = 1
) -> tuple[list[np.ndarray], list[np.ndarray], list[TileInformation]]:
    """Get tiled predictions from a model for the entire dataset.

    Parameters
    ----------
    model : VAEModule
        Lightning model used for prediction.
    dloader : DataLoader
        DataLoader to predict on. Dataset must be a `InMemoryTiledPredDataset` object.
    mmse_count : int, optional
        Number of samples to generate for each input and then to average over for
        MMSE estimation, by default 1.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[TileInformation]]
        Tuple containing:
            - predictions: Predicted images for the dataset.
            - predictions_std: Standard deviation of the predicted images.
            - tiles_info: Information about the tiles coordinates.
    """
    predictions = []
    predictions_std = []
    tiles_info = []
    with torch.no_grad():
        for batch in tqdm(dloader, desc="Predicting patches"):
            inp, tinfo = batch
            inp = inp.cuda()
            tiles_info.extend(tinfo)

            rec_img_list = []
            for _ in range(mmse_count):
                _, out, _ = model(inp)
                
                if model.model.predict_logvar is None:
                    rec_img = out
                    logvar = torch.tensor([-1])
                else:
                    rec_img, logvar = torch.chunk(out, chunks=2, dim=1)
                rec_img_list.append(rec_img.cpu().unsqueeze(0)) # add MMSE dim

            # aggregate results
            samples = torch.cat(rec_img_list, dim=0)
            mmse_imgs = torch.mean(samples, dim=0)
            mmse_std = torch.std(samples, dim=0)
            predictions.append(mmse_imgs.cpu().numpy())
            predictions_std.append(mmse_std.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(predictions_std), tiles_info
