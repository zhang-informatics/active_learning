class Data(object):
    '''
    Data container object to hold features and labels.
    '''
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.__check_args(self.x, self.y)

    def __setattr__(self, name, value):
        if name == 'x' or name == 'y':
            if value is None:
                pass
            else:
                if not hasattr(value, '__iter__'):
                    raise AttributeError("Data must be iterable.")
                if not hasattr(value, 'shape'):
                    raise AttributeError("Data must have a 'shape' attribute.")
        self.__dict__[name] = value    

    def __check_args(self, x, y):
        if x is not None and y is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError("Dimension 0 of x and y do not match. x: {0}, y: {1}"
                                  .format(x.shape, y.shape))

    def __getitem__(self, key):
        x = self.x[key, :]
        y = self.y[key]
        return (x, y)

    def __str__(self):
        x_str = "X: {}".format(self.x)
        y_str = "Y: {}".format(self.y)
        return "{0}\n{1}".format(x_str, y_str)

    def __repr__(self):
        x_str = "X: {}".format(self.x)
        y_str = "Y: {}".format(self.y)
        return "{0}\n{1}\n".format(x_str, y_str)
