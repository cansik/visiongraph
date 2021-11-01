class Vector2:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y
