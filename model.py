from scipy.io import loadmat


class LoadDB:
    internal_time = 0
    internal_temperature = 0

    def __init__(self, discharge, channel, source):
        sawdata = loadmat('saw_data.mat')

        dataType = 'KK3PPF' if source == 'public' else 'KK3JPF'

        self.temperature = sawdata['saw_data'][0, 0][dataType][0, 0]['TE' + str(channel)][0, discharge]
        self.time = sawdata['saw_data'][0, 0][dataType][0, 0]['TIM' + str(channel)][0, discharge]

    @property
    def time(self):
        return self.internal_time

    @time.setter
    def time(self, value):
        self.internal_time = value

    @property
    def temperature(self):
        return self.internal_temperature

    @temperature.setter
    def temperature(self, value):
        self.internal_temperature = value
