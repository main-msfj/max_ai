import typing as t 

class LayerContainerError(Exception):
    """Raised when LayerContainer construction fails validation."""

    def __init__(self, message: str):
        super().__init__(message)

    @classmethod
    def required_one_layer(cls) -> t.Self:
        return cls("LayerContainer requires at least one layer.")
    
    @classmethod
    def duplicated_layers(cls, name: str) -> t.Self:
        return cls(f"Duplicated layer in type in stack: {name}")

