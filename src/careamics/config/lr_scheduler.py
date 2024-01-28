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

from .support.supported_optimizers import SupportedScheduler, get_parameters

class LrSchedulerModel(BaseModel):
    """
    Torch learning rate scheduler.

    Only parameters supported by the corresponding torch lr scheduler will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    Note that mandatory parameters (see the specific LrScheduler signature in the
    link above) must be provided. For example, StepLR requires `step_size`.

    Attributes
    ----------
    name : TorchLRScheduler
        Name of the learning rate scheduler.
    parameters : dict
        Parameters of the learning rate scheduler (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: Literal["ReduceLROnPlateau", "StepLR"]

    # Optional parameters
    parameters: dict = Field(default={}, validate_default=True)

    @field_validator("parameters", mode='before')
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> Dict:
        """
        Validate lr scheduler parameters.

        This method filters out unknown parameters, given the lr scheduler name.

        Parameters
        ----------
        user_params : dict
            Parameters passed on to the torch lr scheduler.
        values : ValidationInfo
            Pydantic field validation info, used to get the lr scheduler name.

        Returns
        -------
        Dict
            Filtered lr scheduler parameters.

        Raises
        ------
        ValueError
            If the lr scheduler name is not specified.
        """
        # None value to default
        if user_params is None:
            user_params = {}
        
        # since we are validating before type validation, enforce is here
        if not isinstance(user_params, dict):
            raise ValueError(
                f"Optimizer parameters must be a dictionary, got {type(user_params)}."
            )
        
        if "name" in values.data:
            lr_scheduler_name = values.data["name"]

            # retrieve the corresponding lr scheduler class
            lr_scheduler_class = getattr(optim.lr_scheduler, lr_scheduler_name)

            # filter the user parameters according to the lr scheduler's signature
            return get_parameters(lr_scheduler_class, user_params)
        else:
            raise ValueError(
                "Cannot validate lr scheduler parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def step_lr_step_size_parameter(cls, lr_scheduler: LrSchedulerModel) -> LrSchedulerModel:
        """
        Check that StepLR lr scheduler has `step_size` parameter specified.

        Parameters
        ----------
        lr_scheduler : LrScheduler
            Lr scheduler to validate.

        Returns
        -------
        LrScheduler
            Validated lr scheduler.

        Raises
        ------
        ValueError
            If the lr scheduler is StepLR and the step_size parameter is not specified.
        """
        if (
            lr_scheduler.name == SupportedScheduler.StepLR
            and "step_size" not in lr_scheduler.parameters
        ):
            raise ValueError(
                "StepLR lr scheduler requires `step_size` parameter, check that it has "
                "correctly been specified in `parameters`."
            )

        return lr_scheduler
