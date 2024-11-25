# Copyright Grantham Taylor.

import builtins
from functools import cache
from enum import Enum


class FieldType(Enum):
    QUANTILES = 1
    CATEGORIES = 2
    TIMESTAMPS = 3


class Registrar(type):
    def __new__(cls, name, bases, class_dict):
        new = super().__new__(cls, name, bases, class_dict)

        if not hasattr(builtins, "tensorfields"):
            builtins.tensorfields = {}

        if not hasattr(builtins, "embedders"):
            builtins.embedders = {}

        if not hasattr(new, "type"):
            return new

        if not isinstance(new.type, FieldType):
            raise ValueError(f"Registered type must be of type FieldType, got {new.type}")

        # this could be cleaner, right?
        parents = list(map(lambda x: str(x.__name__), new.__bases__))

        if "TensorField" in parents:
            builtins.tensorfields[new.type] = new

        if "Embedder" in parents:
            builtins.embedders[new.type] = new

        return new

    @classmethod
    def validate(cls, fieldtype: FieldType) -> bool:
        if fieldtype not in builtins.tensorfields.keys():
            raise KeyError(f"Field type {fieldtype} not registered as tensorfield")

        if fieldtype not in builtins.embedders.keys():
            raise KeyError(f"Field type {fieldtype} not registered as embedder")

    @classmethod
    @cache
    def tensorfield(cls, fieldtype: FieldType) -> "TensorField":
        cls.validate(fieldtype)
        return builtins.tensorfields[fieldtype]

    @classmethod
    @cache
    def embedders(cls, fieldtype: FieldType) -> "Embedder":
        cls.validate(fieldtype)
        return builtins.embedders[fieldtype]


class TensorField(metaclass=Registrar):
    pass


class Embedder(metaclass=Registrar):
    pass


class MyTensorField(TensorField):
    type = FieldType.QUANTILES


class MyOtherTensorField(TensorField):
    type = FieldType.CATEGORIES


class MyEmbedder(Embedder):
    type = FieldType.QUANTILES


class MyOtherEmbedder(Embedder):
    type = FieldType.CATEGORIES
