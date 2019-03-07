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
    internal_shots = 0
    internal_time = 0
    internal_temperature = 0
    internal_channels = 0
    internal_channels_model = 0
    internal_psi_r = 0
    internal_q_database = 0

    def load(self):
        """ -----------------------------------------
             version: 0.10
             desc: load data from matlab
         ----------------------------------------- """
        sawdata = loadmat('saw_data.mat')
        self.shots = sawdata['saw_data'][0, 0]['SHOT'].ravel()

        return sawdata

    def assign(self, db, discharge, source):
        """ -----------------------------------------
             version: 0.10
             desc: assign certain discharge data from preloaded DB
         ----------------------------------------- """

        sawdata = db

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
        self.psi_r = sawdata['saw_data'][0, 0]['PSIPROFILE'][0, 0]['R'][0][discharge].ravel()
        self.q_database = sawdata['saw_data'][0, 0]['QPROFILE'][0, 0]['QPRE'][0, discharge].ravel()

    @property
    def q_database(self):
        return self.internal_q_database

    @q_database.setter
    def q_database(self, value):
        self.internal_q_database = value

    @property
    def channels(self):
        return self.internal_channels

    @channels.setter
    def channels(self, value):
        self.internal_channels = value

    @property
    def psi_r(self):
        return self.internal_psi_r

    @psi_r.setter
    def psi_r(self, value):
        self.internal_psi_r = value

    @property
    def channels_model(self):
        return self.internal_channels_model

    @channels_model.setter
    def channels_model(self, value):
        self.internal_channels_model = value

    @property
    def shots(self):
        return self.internal_shots

    @shots.setter
    def shots(self, value):
        self.internal_shots = value

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
