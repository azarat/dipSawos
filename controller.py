import model as DBModel
from math import fabs
import numpy as np

class dataPreparing:


    def setNewTime(self, newTime):
        self.t = newTime

    def setNewTemp(self, newTemp):
        self.temperature = newTemp


    def filtering(self, time, temp):

        newTime = []
        newTemp = []
        numOfSteps = 10
        stepW = (max(time) - min(time)) / numOfSteps
        for fixT in range(0, numOfSteps):

            newTime = np.arange(min(time), max(time), stepW)
            integral = 0

            for index, t in enumerate(time):
                integral += temp[index] * (1 - fabs( (t - ((stepW - 1) / 2)) / (stepW / 2)))

            newTemp.append(integral)

        self.setNewTime(newTime)
        self.setNewTemp(newTemp)
        # return 1

    def getData(self, discharge, channel, type):
        DB = DBModel.loadDB()
        DB.loadData(discharge, channel, type)
        time = DB.t
        temp = DB.temperature

        self.filtering(time, temp)


