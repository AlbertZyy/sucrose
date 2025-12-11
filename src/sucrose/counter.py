
class Buffer():
    _value : int
    _buffer : int

    def __init__(self, value: int = 0, buffer: int = 0):
        self._value = value
        self._buffer = buffer

    def __repr__(self):
        return "Int({} <- {})".format(self._value, self._buffer)

    def __int__(self):
        return self._value

    def __float__(self):
        return float(self._value)

    def __index__(self):
        return self._value

    @property
    def buffer(self):
        return self._buffer

    @property
    def value(self):
        return self._value

    def step(self, n: int, /, interval: int):
        tmp_val = self._buffer + n
        self._buffer = tmp_val % interval

        if tmp_val >= interval:
            self._value += tmp_val - self._buffer
            return True

        return False

    def update(self):
        self._value += self._buffer
        self._buffer = 0

    def clean(self):
        self._buffer = 0
