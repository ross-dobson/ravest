class Parameter():

    def __init__(self, value: float, unit: str, fixed=False):
        """
        Initialize a parameter object.

        Parameters
        ----------
        value : float
            The value of the parameter.
        unit : str
            The unit of measurement for the parameter. This is only used for display purposes.
        fixed : bool
            Indicates whether the parameter is fixed or free to vary in fitting. Default is False.
        """
        self.value = value
        self.unit = unit 
        self.fixed = fixed

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(value={self.value!r}, unit={self.unit!r}, fixed={self.fixed!r})"

    def __str__(self):
        class_name = type(self).__name__
        return f"{class_name} {self.value} {self.unit}"

