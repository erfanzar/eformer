# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import dataclasses
import difflib
import inspect
import typing as tp
from dataclasses import fields

import jax
import optax

from ._config import (
    AdafactorConfig,
    AdamWConfig,
    KronConfig,
    LionConfig,
    MarsConfig,
    MuonConfig,
    RMSPropConfig,
    SchedulerConfig,
    SerializationMixin,
    SoapConfig,
    WhiteKronConfig,
)
from ._custom import mars, white_kron
from ._tx import optax_add_scheduled_weight_decay


class SchedulerFactory:
    """
    Factory class for creating learning rate schedulers.

    This class provides methods to create schedulers based on a configuration object (`SchedulerConfig`).
    It supports linear and cosine schedulers with optional warmup steps.

    Methods:
        create_scheduler: Creates a scheduler based on the provided configuration.
        _create_linear: Creates a linear scheduler with optional warmup.
        _create_cosine: Creates a cosine scheduler with optional warmup.
    """

    @staticmethod
    def create_scheduler(
        config: SchedulerConfig,
        custom_scheduler: tp.Callable[[int], optax.Schedule] | None = None,
    ) -> optax.Schedule:
        """
        Create a scheduler based on the provided configuration.

        Args:
            config (SchedulerConfig): Configuration object for the scheduler.
            custom_scheduler (Optional[Callable[[int], optax.Schedule]]): Custom scheduler function. Defaults to None.

        Returns:
            optax.Schedule: The created scheduler.

        Raises:
            ValueError: If the configuration is invalid or unsupported scheduler type is provided.
        """
        if custom_scheduler is not None:
            if config.steps is None:
                raise ValueError("Custom schedulers require steps configuration")
            return custom_scheduler(config.steps)
        if config.scheduler_type is None:
            return optax.constant_schedule(config.learning_rate)
        if config.steps is None:
            raise ValueError("Steps must be specified for configured schedulers")
        if config.scheduler_type == "linear":
            return SchedulerFactory._create_linear(config)
        elif config.scheduler_type == "cosine":
            return SchedulerFactory._create_cosine(config)
        else:
            raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")

    @staticmethod
    def _create_linear(config: SchedulerConfig) -> optax.Schedule:
        """
        Create a linear scheduler with optional warmup.

        Args:
            config (SchedulerConfig): Configuration object for the scheduler.

        Returns:
            optax.Schedule: The created linear scheduler.
        """
        base_scheduler = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=config.learning_rate_end,
            transition_steps=config.steps,
        )

        if config.warmup_steps:
            warmup = optax.linear_schedule(
                init_value=1e-8,
                end_value=config.learning_rate,
                transition_steps=config.warmup_steps,
            )
            return optax.join_schedules(
                schedules=[warmup, base_scheduler],
                boundaries=[config.warmup_steps],
            )
        return base_scheduler

    @staticmethod
    def _create_cosine(config: SchedulerConfig) -> optax.Schedule:
        """
        Create a cosine scheduler with optional warmup.

        Args:
            config (SchedulerConfig): Configuration object for the scheduler.

        Returns:
            optax.Schedule: The created cosine scheduler.
        """
        if config.warmup_steps:
            return optax.warmup_cosine_decay_schedule(
                init_value=1e-8,
                peak_value=config.learning_rate,
                warmup_steps=config.warmup_steps,
                decay_steps=config.steps - config.warmup_steps,
                end_value=config.learning_rate_end or 0.0,
                exponent=config.exponent,
            )
        return optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.steps,
            alpha=config.learning_rate_end or 0.0,
        )


class OptimizerFactory:
    """
    Factory class for creating optimizers with validated configurations.

    This class provides methods to create optimizers based on a configuration object.
    It supports multiple optimizer types, including Adafactor, AdamW, Lion, and RMSProp.

    Attributes:
        _OPTIMIZER_REGISTRY (Dict[str, Tuple[Type, Type]]): Registry of supported optimizers and their configurations.

    Methods:
        register_optimizer: Registers a new optimizer type.
        create: Creates an optimizer with validated configuration.
        _convert_dtypes: Converts string dtype representations to JAX dtypes.
        _validate_kwargs: Validates additional parameters for the optimizer.
        _build_optimizer: Constructs the final optimizer chain.
        generate_template: Generates a configuration template for the specified optimizer.
        serialize_config: Serializes configuration to different formats.
        deserialize_config: Deserializes configuration from different formats.
    """

    _OPTIMIZER_REGISTRY: tp.ClassVar = {
        "adafactor": (optax.adafactor, AdafactorConfig),
        "adamw": (optax.adamw, AdamWConfig),
        "lion": (optax.lion, LionConfig),
        "mars": (mars, MarsConfig),
        "muon": (optax.contrib.muon, MuonConfig),
        "rmsprop": (optax.rmsprop, RMSPropConfig),
        "skew": (white_kron.skew, WhiteKronConfig),
        "quad": (white_kron.quad, WhiteKronConfig),
    }

    @classmethod
    def register_optimizer(cls, name: str, optimizer_cls: type, config_cls: type):
        """
        Register a new optimizer type.

        Args:
            name (str): Name of the optimizer.
            optimizer_cls (Type): Optimizer class.
            config_cls (Type): Configuration class for the optimizer.
        """
        cls._OPTIMIZER_REGISTRY[name] = (optimizer_cls, config_cls)

    @classmethod
    def create(
        cls,
        optimizer_type: str,
        scheduler_config: SchedulerConfig,
        optimizer_config: AdafactorConfig
        | AdamWConfig
        | KronConfig
        | LionConfig
        | MarsConfig
        | MuonConfig
        | RMSPropConfig
        | SoapConfig
        | WhiteKronConfig = None,
        *,
        weight_decay: float = 0.0,
        weight_decay_mask: tp.Any | None = None,
        gradient_accumulation_steps: int = 1,
        clip_grad: float | None = None,
        custom_scheduler: tp.Callable[[int], optax.Schedule] | None = None,
        **kwargs,
    ) -> tuple[optax.GradientTransformation, optax.Schedule]:
        """
        Create an optimizer with validated configuration.

        Args:
            optimizer_type (str): One of the registered optimizer types.
            scheduler_config (SchedulerConfig): Configured scheduler parameters.
            optimizer_config (Union[AdafactorConfig, AdamWConfig, LionConfig, MuonConfig, RMSPropConfig]):
                Optimizer-specific configuration.
            weight_decay (float): Global weight decay rate. Defaults to 0.0.
            weight_decay_mask (Optional[Any]): Mask for weight decay application. Defaults to None.
            gradient_accumulation_steps (int): Steps for gradient accumulation. Defaults to 1.
            clip_grad (Optional[float]): Global clip gradient norm value. Defaults to None.
            custom_scheduler (Optional[Callable[[int], optax.Schedule]]): Optional custom scheduler function.
                Defaults to None.
            **kwargs: Additional optimizer-specific parameters.

        Returns:
            Tuple[optax.GradientTransformation, optax.Schedule]: A tuple containing the optimizer and scheduler.

        Raises:
            ValueError: If the optimizer type is unsupported or the configuration is invalid.
            TypeError: If the configuration type is invalid.
        """

        if optimizer_type not in cls._OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_type}. Available: {list(cls._OPTIMIZER_REGISTRY.keys())}"
            )

        if optimizer_config is None:
            optimizer_config = cls._OPTIMIZER_REGISTRY[optimizer_type][1]()
            for key in list(kwargs.keys()):
                if key in inspect.signature(optimizer_config.__class__).parameters:
                    setattr(optimizer_config, key, kwargs.pop(key))
        cls._convert_dtypes(optimizer_config)
        optimizer_cls, config_cls = cls._OPTIMIZER_REGISTRY[optimizer_type]

        if not isinstance(optimizer_config, config_cls):
            raise TypeError(
                f"Invalid config type {type(optimizer_config)} for optimizer {optimizer_type}. Expected {config_cls}"
            )

        if scheduler_config.scheduler_type is None and scheduler_config.warmup_steps:
            raise ValueError("Warmup steps require specifying a scheduler type")

        scheduler = SchedulerFactory.create_scheduler(
            scheduler_config,
            custom_scheduler,
        )

        return cls._build_optimizer(
            optimizer_cls=optimizer_cls,
            optimizer_config=optimizer_config,
            scheduler=scheduler,
            weight_decay=weight_decay,
            weight_decay_mask=weight_decay_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clip_grad=clip_grad,
            **kwargs,
        )

    @staticmethod
    def _convert_dtypes(config: tp.Any):
        """
        Automatically convert string dtype representations to JAX dtypes.

        Args:
            config (Any): Configuration object.

        Raises:
            ValueError: If an invalid dtype is specified.
        """
        for field in fields(config):
            if "dtype" in field.name and isinstance(getattr(config, field.name), str):
                dtype = getattr(jax.numpy, getattr(config, field.name), None)
                if dtype is None:
                    raise ValueError(f"Invalid dtype specified: {getattr(config, field.name)}")
                setattr(config, field.name, dtype)

    @classmethod
    def _validate_kwargs(cls, config: tp.Any, kwargs: dict[str, tp.Any]):
        """
        Validate additional parameters with helpful error messages.

        Args:
            config (Any): Configuration object.
            kwargs (Dict[str, Any]): Additional parameters.

        Raises:
            ValueError: If unexpected parameters are provided.
        """
        valid_params = inspect.signature(config.__class__).parameters
        for kwarg in kwargs:
            if kwarg not in valid_params:
                suggestions = ", ".join(difflib.get_close_matches(kwarg, valid_params.keys()))
                msg = (
                    f"Unexpected parameter '{kwarg}' for {config.__class__.__name__}. "
                    f"Valid parameters: {list(valid_params.keys())}"
                )
                if suggestions:
                    msg += f". Did you mean: {suggestions}?"
                raise ValueError(msg)

    @classmethod
    def _build_optimizer(
        cls,
        optimizer_cls: type,
        optimizer_config: tp.Any,
        scheduler: optax.Schedule,
        **common_kwargs,
    ) -> tuple[optax.GradientTransformation, optax.Schedule]:
        """
        Construct the final optimizer chain.

        Args:
            optimizer_cls (Type): Optimizer class.
            optimizer_config (Any): Optimizer configuration.
            scheduler (optax.Schedule): Learning rate scheduler.
            **common_kwargs: Common keyword arguments.

        Returns:
            Tuple[optax.GradientTransformation, optax.Schedule]: A tuple containing the optimizer and scheduler.
        """

        config_dict = {k: v for k, v in vars(optimizer_config).items() if not k.startswith("_")}

        optimizer = optimizer_cls(learning_rate=scheduler, **config_dict)

        chain = []

        if common_kwargs.get("clip_grad"):
            chain.append(optax.clip_by_global_norm(common_kwargs["clip_grad"]))

        chain.append(optimizer)

        if common_kwargs["weight_decay"] != 0.0:
            chain.append(
                optax_add_scheduled_weight_decay(
                    lambda step: -scheduler(step) * common_kwargs["weight_decay"],
                    common_kwargs["weight_decay_mask"],
                )
            )

        tx = optax.chain(*chain)

        if common_kwargs["gradient_accumulation_steps"] > 1:
            tx = optax.MultiSteps(tx, common_kwargs["gradient_accumulation_steps"])

        return tx, scheduler

    @classmethod
    def generate_template(cls, optimizer_type: str) -> str:
        """
        Generate a configuration template for the specified optimizer.

        Args:
            optimizer_type (str): Name of the optimizer.

        Returns:
            str: Configuration template.

        Raises:
            ValueError: If the optimizer type is unknown.
        """
        try:
            config_cls = cls._OPTIMIZER_REGISTRY[optimizer_type][1]
        except KeyError:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}") from None

        fields = []
        for field in dataclasses.fields(config_cls):
            field_type = tp.get_type_hints(config_cls)[field.name]
            default = f" = {field.default}" if not isinstance(field.default, dataclasses._MISSING_TYPE) else ""

            if hasattr(field_type, "__name__"):
                type_name = field_type.__name__
            else:
                type_name = str(field_type)

            fields.append(f"    {field.name}: {type_name}{default}")

        return f"{config_cls.__name__}(\n" + "\n".join(fields) + "\n)"

    @classmethod
    def serialize_config(
        cls,
        config: SerializationMixin,
        format: str = "dict",  # noqa:A002
    ) -> dict | str:
        """
        Serialize configuration to different formats.

        Args:
            config (SerializationMixin): Configuration object.
            format (str): Serialization format. Supported formats: 'dict', 'json'.

        Returns:
            Union[Dict, str]: Serialized configuration.

        Raises:
            ValueError: If the format is unsupported.
        """
        if format not in ["dict", "json"]:
            raise ValueError("Supported formats: 'dict', 'json'")

        if format == "dict":
            return config.to_dict()
        return config.to_json()

    @classmethod
    def deserialize_config(
        cls,
        optimizer_type: str,
        data: dict | str,
        format: str = "dict",  # noqa:A002
    ) -> SerializationMixin:
        """
        Deserialize configuration from different formats.

        Args:
            optimizer_type (str): Name of the optimizer.
            data (Union[Dict, str]): Serialized configuration data.
            format (str): Serialization format. Supported formats: 'dict', 'json'.

        Returns:
            SerializationMixin: Deserialized configuration object.

        Raises:
            ValueError: If the optimizer type is unknown or the format is unsupported.
            TypeError: If the input data type is invalid.
        """
        try:
            config_cls = cls._OPTIMIZER_REGISTRY[optimizer_type][1]
        except KeyError:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}") from None

        if format == "json":
            if not isinstance(data, str):
                raise TypeError("Expected string input for JSON format")
            return config_cls.from_json(data)

        if format == "dict":
            if not isinstance(data, dict):
                raise TypeError("Expected dictionary input for dict format")
            return config_cls.from_dict(data)

        raise ValueError("Unsupported format")
