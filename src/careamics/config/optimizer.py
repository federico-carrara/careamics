from __future__ import annotations
from typing import Dict, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
    field_validator
)
from torch import optim

from .support.supported_optimizers import SupportedOptimizer, get_parameters


class OptimizerModel(BaseModel):
    """
    Torch optimizer.

    Only parameters supported by the corresponding torch optimizer will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#algorithms

    Note that mandatory parameters (see the specific Optimizer signature in the
    link above) must be provided. For example, SGD requires `lr`.

    Attributes
    ----------
    name : TorchOptimizer
        Name of the optimizer.
    parameters : dict
        Parameters of the optimizer (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: Literal["Adam", "SGD"]

    # Optional parameters, empty dict default value to allow filtering dictionary
    parameters: dict = Field(default={}, validate_default=True)

    @field_validator("parameters", mode='before')
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> Dict:
        """
        Validate optimizer parameters.

        This method filters out unknown parameters, given the optimizer name.

        Parameters
        ----------
        user_params : dict
            Parameters passed on to the torch optimizer.
        values : ValidationInfo
            Pydantic field validation info, used to get the optimizer name.

        Returns
        -------
        Dict
            Filtered optimizer parameters.

        Raises
        ------
        ValueError
            If the optimizer name is not specified.
        """
        # None value to default
        if user_params is None:
            user_params = {}

        # since we are validating before type validation, enforce is here
        if not isinstance(user_params, dict):
            raise ValueError(
                f"Optimizer parameters must be a dictionary, got {type(user_params)}."
            )

        if "name" in values.data: # TODO test if that clause if really necessary
            optimizer_name = values.data["name"]

            # retrieve the corresponding optimizer class
            optimizer_class = getattr(optim, optimizer_name)

            # filter the user parameters according to the optimizer's signature
            return get_parameters(optimizer_class, user_params)
        else:
            raise ValueError(
                "Cannot validate optimizer parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def sgd_lr_parameter(cls, optimizer: OptimizerModel) -> OptimizerModel:
        """
        Check that SGD optimizer has the mandatory `lr` parameter specified.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer to validate.

        Returns
        -------
        Optimizer
            Validated optimizer.

        Raises
        ------
        ValueError
            If the optimizer is SGD and the lr parameter is not specified.
        """
        if optimizer.name == SupportedOptimizer.SGD and "lr" not in optimizer.parameters:
            raise ValueError(
                "SGD optimizer requires `lr` parameter, check that it has correctly "
                "been specified in `parameters`."
            )

        return optimizer
