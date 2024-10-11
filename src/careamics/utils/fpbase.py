"""Utilities for querying FPBase.

Adapted from https://github.com/tlambert03/microsim. 
"""

import json
from typing import Optional, Sequence, Union
from functools import cached_property
from urllib.request import Request, urlopen

from pydantic import BaseModel, field_validator, model_validator
import torch

from .spectral import Spectrum

FPBASE_URL = "https://www.fpbase.org/graphql/"


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