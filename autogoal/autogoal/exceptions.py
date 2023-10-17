class InterfaceIncompatibleError(ValueError):
    def __init__(self, cls) -> None:
        super().__init__(f"Cannot find compatible implementations for <class {cls}>")
        self.cls = cls
