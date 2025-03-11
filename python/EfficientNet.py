

class EfficientNet:
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate