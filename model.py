from scipy.io import loadmat


class LoadDB:
    internal_time = 0
    internal_temperature = 0
    internal_channels = 0

    def __init__(self, discharge, channel, source):
        sawdata = loadmat('saw_data.mat')

        dataType = 'KK3PPF' if source == 'public' else 'KK3JPF'

        channels_temp = {}

        for i in range(1, 97):
            channels_temp.update({
                i: sawdata['saw_data'][0, 0]['KK3PPF'][0,0]['RA' + str("{:0>2d}".format(i))][0, discharge]
            })
        self.channels = channels_temp

        if type(channel) is tuple:

            temperature_temp = {}
            time_temp = {}

            for i in range(channel[0], channel[1]):
                temperature_temp.update({
                    i: sawdata['saw_data'][0, 0][dataType][0, 0]['TE' + str("{:0>2d}".format(i))][0, discharge].ravel()
                })
                time_temp.update({
                    i: sawdata['saw_data'][0, 0][dataType][0, 0]['TIM' + str("{:0>2d}".format(i))][0, discharge].ravel()
                })

            self.temperature = temperature_temp
            self.time = time_temp

        else:
            self.temperature = sawdata['saw_data'][0, 0][dataType][0, 0]['TE' + str(channel)][0, discharge]
            self.time = sawdata['saw_data'][0, 0][dataType][0, 0]['TIM' + str(channel)][0, discharge]

    @property
    def channels(self):
        return self.internal_channels

    @channels.setter
    def channels(self, value):
        self.internal_channels = value

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
