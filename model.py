from scipy.io import loadmat

class loadDB:

    def setTime(self, matLabTime):
        self.t = matLabTime

    def setTemperature(self, matLabTemperature):
        self.temperature = matLabTemperature

    def loadData(self, discharge, channel, source):
        sawdata = loadmat('saw_data.mat')

        dataType = 'KK3PPF' if source == 'public' else 'KK3JPF'

        self.setTemperature(sawdata['saw_data'][0, 0][dataType][0, 0]['TE' + str(channel)][0, discharge])
        self.setTime(sawdata['saw_data'][0, 0][dataType][0, 0]['TIM' + str(channel)][0, discharge])
