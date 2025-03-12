"""Type used to represent all transformations users can create."""

from typing import Union

from pydantic import Discriminator
from typing_extensions import Annotated

from .n2v_manipulate_model import N2VManipulateModel
from .normalize_model import NormalizeModel
from .standardize_model import StandardizeModel
from .xy_flip_model import XYFlipModel
from .xy_random_rotate90_model import XYRandomRotate90Model

TRANSFORMS_UNION = Annotated[
    Union[
        NormalizeModel,
        StandardizeModel,
        XYFlipModel,
        XYRandomRotate90Model,
        N2VManipulateModel,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""Available transforms in CAREamics."""
