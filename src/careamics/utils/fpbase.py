"""Utilities for querying FPBase.

Taken from `microsim` library. 
"""

import json
from typing import Optional, Union
# from functools import cache
from urllib.request import Request, urlopen
import warnings

from pydantic import BaseModel, field_validator, model_validator
import numpy as np
import torch

FPBASE_URL = "https://www.fpbase.org/graphql/"

class Spectrum(BaseModel):
    """A Spectrum object defined by its intensity and wavelengths.
    
    Adapted from https://github.com/tlambert03/microsim.
    """
    wavelength: torch.Tensor
    """The set of wavelength of the spectrum."""
    intensity: torch.Tensor
    """The intensity of the spectrum."""
    
    @field_validator("intensity", mode="after")
    @classmethod
    def _validate_intensity(cls, value: torch.Tensor) -> torch.Tensor:
        if not np.all(value >= 0):
            warnings.warn(
                "Clipping negative intensity values in spectrum to 0", stacklevel=2
            )
            value = np.clip(value, 0, None)
        return value

    @model_validator(mode="before")
    def _validate_shapes(cls, value: 'Spectrum') -> 'Spectrum':
        if "wavelength" in value and "intensity" in value:
            if not len(value["wavelength"]) == len(value["intensity"]):
                raise ValueError(
                    "Wavelength and intensity must have the same length"
                )
        return value


# @cache
def get_dye_by_id(id: Union[str, int]) -> dict:
    """Given a FPBase dye ID, return the dye's information.
    
    Parameters
    ----------
    id : Union[str, int]
        The dye ID on FPBase.
    """
    query = """
    {{
        dye(id: {id}) {{
            name
            id
            exMax
            emMax
            extCoeff
            qy
            spectra {{ subtype data }}
        }}
    }}
    """
    resp = _fpbase_query(query.format(id=id))
    return json.loads(resp)["data"]["dye"]


# @cache
def get_protein_by_id(id: str) -> dict:
    """Given a FPBase protein ID, return the protein's information.
    
    Parameters
    ----------
    id : str
        The protein ID  on FPBase.
    """
    query = """
    {{
        protein(id: "{id}") {{
            name
            id
            states {{
                id
                name
                exMax
                emMax
                extCoeff
                qy
                lifetime
                spectra {{ subtype data }}
            }}
            defaultState {{
                id
            }}
        }}
    }}
    """
    resp = _fpbase_query(query.format(id=id))
    return json.loads(resp)["data"]["protein"]


def _fpbase_query(query: str) -> bytes:
    """Query FPBase to get FP data.
    
    Parameters
    ----------
    query : str
        The query.
    """
    headers = {"Content-Type": "application/json", "User-Agent": "careamics"}
    data = json.dumps({"query": query}).encode("utf-8")
    req = Request(FPBASE_URL, data=data, headers=headers)
    with urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}")
        return response.read()  # type: ignore


# @cache
def fluorophore_ids() -> dict[str, dict[str, str]]:
    """Return a lookup table of fluorophore {name: {id: ..., type: ...}}.
    
    Returns
    -------
    dict[str, dict[str, str]]
        A lookup table with the 'name', 'slug' and 'id' of all existing FPs on FPBase.
    """
    resp = _fpbase_query("{ dyes { id name slug } proteins { id name slug } }")
    data: dict[str, list[dict[str, str]]] = json.loads(resp)["data"]
    lookup: dict[str, dict[str, str]] = {}
    for key in ["dyes", "proteins"]:
        for item in data[key]:
            lookup[item["name"].lower()] = {"id": item["id"], "type": key[0]}
            lookup[item["slug"]] = {"id": item["id"], "type": key[0]}
            if key == "proteins":
                lookup[item["id"]] = {"id": item["id"], "type": key[0]}
    return lookup


# @cache
def get_fluorophore(name: str) -> dict:
    """Get the fluorophore information from FPBase given its name.
    
    Parameters
    ----------
    name : str
        The name of the fluorophore.
        
    Returns
    -------
    dict
        The fluorophore information.
    """
    _ids = fluorophore_ids()
    fluor_info = _ids[name.lower()]
    if fluor_info["type"] == "d":
        return get_dye_by_id(fluor_info["id"])
    elif fluor_info["type"] == "p":
        return get_protein_by_id(fluor_info["id"])
    raise ValueError(f"Invalid fluorophore type {fluor_info['type']!r}")


def get_fp_emission_spectrum(name: str) -> Optional[Spectrum]:
    """Get the fluorophore emission spectrum (wavelengths and intensities).
    
    The FP information is scraped from FPBase given its name.
    Then spectral information is extracted and returned as a `Spectrum` object.
    
    NOTE: in FPBase, intensities are normalized s.t. the max is 1.
    
    Parameters
    ----------
    name : str
        The name of the fluorophore.
        
    Returns
    -------
    Optional[torch.Tensor]
        The emission spectrum of the fluorophore. Shape is [W, 2].
    """
    flurophore = get_fluorophore(name)
    state = next(
        st for st in flurophore["states"] 
        if st["id"] == flurophore["defaultState"]["id"]
    )
    spectrum = next(
        (sp["data"] for sp in state["spectra"] if sp["subtype"] == "EM"), None
    )
    spectrum = torch.tensor(spectrum) if spectrum is not None else None
    return Spectrum(wavelength=spectrum[:, 0], intensity=spectrum[:, 1])


class FPRefMatrix(BaseModel):
    """
    This class is used to create a reference matrix of fluorophore emission spectra
    for the spectral unmixing task.
    """
    fp_names: Optional[list[str]] = None
    w_bins: int = 32
    fp_em_list = None # list of xr.DataArray's containing emission spectra
    
    @property
    def N(self) -> int:  # TODO: check consistency with naming
        return len(self.fp_names)

    @property
    def P(self) -> int:  # TODO: check consistency with naming
        return len(self.w_bins)
    
    @property
    def sbins(self) -> int:
        return sorted(set([bins[0] for bins in self.w_bins] + [self.w_bins[-1][1]]))
    
    @property
    def fp_spectra(self) -> list[[torch.Tensor]]:
        pass
    
    def _fetch_fp_s(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [Fluorophore.from_fpbase(name=fp_name) for fp_name in self.fp_names]
    
    def _normalize(self) -> np.ndarray:
        assert self.fp_em_list is not None
        return [
            (fp_em - fp_em.min()) / (fp_em.max() - fp_em.min())
            for fp_em in self.fp_em_list
        ]
    
    def _fill_NaNs(self, num: int = 0) -> list[xr.DataArray]:
        assert self.fp_em_list is not None
        return [
            fp_em.fillna(num)
            for fp_em in self.fp_em_list
        ]
    
    def _bin_spectra(self) -> list[xr.DataArray]:
        assert self.fp_em_list is not None
        return [
            fp_em.groupby_bins(fp_em["w"], self.sbins).sum()
            for fp_em in self.fp_em_list
        ]
        
    def create(self) -> np.ndarray:
        self.fp_list = self._fetch_FPs()
        self.fp_em_list = [
            xr.DataArray(
                fp.emission_spectrum.intensity, 
                coords=[fp.emission_spectrum.wavelength.magnitude], 
                dims=["w"]
            )
            for fp in self.fp_list
        ]
        self.fp_em_list = self._bin_spectra()
        self.fp_em_list = self._fill_NaNs()
        self.fp_em_list = self._normalize()
        return np.stack(
            [fp_em.values for fp_em in self.fp_em_list], 
            axis=1
        )