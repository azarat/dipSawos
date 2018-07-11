from scipy.io import loadmat


class Model:

    @staticmethod
    def load(filename):
        """ -----------------------------------------
             version: 0.3
             desc: load data from matlab
             :param filename: string val
             :return mat raw type ?
         ----------------------------------------- """

        data = loadmat(filename)
        return data


class LoadDB:
    internal_time = 0
    internal_temperature = 0
    internal_channels = 0
    internal_channels_model = 0

    def __init__(self, discharge, source):
        sawdata = loadmat('saw_data.mat')

        data_type = 'KK3PPF' if source == 'public' else 'KK3JPF'

        """ For some reason we need to have model/standard to sort channels """
        channels_temp = {}
        for i in range(1, 97):
            channels_temp.update({
                i: sawdata['saw_data'][0, 0]['KK3PPF'][0,0]['RA' + str("{:0>2d}".format(i))][0, 25]
            })
        self.channels_model = channels_temp

        channels_temp = {}
        for i in range(1, 97):
            channels_temp.update({
                i: sawdata['saw_data'][0, 0]['KK3PPF'][0,0]['RA' + str("{:0>2d}".format(i))][0, discharge]
            })
        self.channels = channels_temp

        temperature_temp = {}
        for i in range(1, 97):
            temperature_temp.update({
                i: sawdata['saw_data'][0, 0][data_type][0, 0]['TE' + str("{:0>2d}".format(i))][0, discharge].ravel()
            })

        self.time = sawdata['saw_data'][0, 0][data_type][0, 0]['TIM01'][0, discharge].ravel()
        self.temperature = temperature_temp

    @property
    def channels(self):
        return self.internal_channels

    @channels.setter
    def channels(self, value):
        self.internal_channels = value

    @property
    def channels_model(self):
        return self.internal_channels_model

    @channels_model.setter
    def channels_model(self, value):
        self.internal_channels_model = value

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
