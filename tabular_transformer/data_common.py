import random
import pandas as pd
from typing import Dict, Optional, Union
import sys
from ast import literal_eval
from dataclasses import asdict, fields
from typing import Literal, get_type_hints
from pathlib import Path


class TypeCheckMeta(type):
    def __call__(cls, *args, **kwargs):
        # Check if any positional arguments are passed
        if args:
            raise TypeError(f"{cls} only accepts keyword arguments.")

        # Get the field definitions with type hints
        field_defs = {f.name: f.type for f in fields(cls)}

        all_args = {**kwargs}

        # Perform type checking
        for name, val in all_args.items():
            expect_type = field_defs.get(name)
            assert expect_type is not None, f"bad {cls} arguments `{name}`"
            if expect_type is float and isinstance(val, int):
                val = float(val)
            # Special case for Literal
            if hasattr(expect_type, "__origin__") and expect_type.__origin__ is Literal:
                assert val in expect_type.__args__ and isinstance(
                    val, type(expect_type.__args__[0])
                ), f"{val} not in {expect_type.__args__}."
            elif hasattr(expect_type, "__origin__") and expect_type.__origin__ is dict:
                assert all(isinstance(key, expect_type.__args__[0]) and isinstance(
                    value, expect_type.__args__[1]) for key, value in val.items()
                ), f"{val} must be type {expect_type}"
            else:
                assert isinstance(
                    val, expect_type
                ), f"{cls} init parameter type mismatch, key: ({name}) expect type: {expect_type}, pass value: {val}"

        # Call the original __init__ method
        return super().__call__(*args, **kwargs)


class DataclassTool(metaclass=TypeCheckMeta):
    def __init__(self):
        raise NotImplementedError("DataclassTool should not be instantiated.")

    def update(self, hypara: str, val):
        if hypara in asdict(self):
            # ensure the types match
            expect_type = get_type_hints(self)[hypara]

            if expect_type is float and isinstance(val, int):
                val = float(val)

            if expect_type is bool and isinstance(val, str):
                if val.lower() == "false":
                    val = False
                elif val.lower() == "true":
                    val = True

            # Special case for Literal
            if hasattr(expect_type, "__origin__") and expect_type.__origin__ is Literal:
                assert val in expect_type.__args__ and isinstance(
                    val, type(expect_type.__args__[0])
                ), f"{val} not in {expect_type.__args__}."
            else:
                assert isinstance(
                    val, expect_type
                ), f"hyperparameter type mismatch, key: ({hypara}) expect type: {expect_type}, pass value: {val}"

            print(f"Overriding hyperparameter: {hypara} = {val}")
            setattr(self, hypara, val)
        else:
            raise ValueError(f"Unknown config hyperparameter key: {hypara}")

    def __str__(self):
        return f"{type(self).__name__}: {asdict(self)}"

    def asdict(self):
        return asdict(self)

    def config_from_cli(self):
        for arg in sys.argv[1:]:
            # assume it's a --key=value argument
            assert arg.startswith(
                '--'), f"specify hyperparameters must in --key=value format"
            key, val = arg.split('=')
            key = key[2:]  # skip --

            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val

            self.update(key, attempt)
