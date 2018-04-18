import model as DBModel
from math import fabs, floor
import numpy as np
from scipy import signal


class DataPreparing:
    internal_time_original = 0
    internal_temperature_original = 0
    internal_time = 0
    internal_temperature = 0

    def filtering_custom(self, time, temp, windowW):
        """ !!! Should be removed but I don't want XD """

        """ Custom window filtering (problem with output Y scale) """
        newTemp = []
        numOfSteps = windowW  # WRONG: windowW should be discreteStepW
        discreteStepW = int(floor(len(time) / numOfSteps))
        stepW = (max(time) - min(time)) / numOfSteps
        newTime = np.arange(min(time), max(time), stepW)

        for fixT in range(0, numOfSteps):
            sumT = 0

            for index in range(fixT * discreteStepW, (fixT * discreteStepW) + discreteStepW):
                # sumT += (temp[index] * 27 * 1000) * (1 - fabs( ((index - fixT * discreteStepW) - ((discreteStepW - 1) / 2)) / (discreteStepW / 2)))
                sumT += (temp[index] * 27 * 1000) * (1 - pow((((2 * (index - fixT * discreteStepW)) / (discreteStepW - 1)) - 1), 2))

            integral = sumT / discreteStepW

            newTemp.append(integral)

    def filtering_single(self, time, temp, windowW):
        """ !!! Should be removed but I don't want XD """

        """ Window filtering with single window """
        sig = temp.ravel() * 27 * 1000
        win = signal.triang(windowW)
        newTemp = signal.convolve(sig, win, mode='valid') / sum(win)
        newTime = time[0:(len(time) - windowW + 1)]

        self.time = newTime
        self.temperature = newTemp

    def filtering_multiple(self, time, temp, windowW):

        """ Window filtering with multiple windows """
        newTime = time[0:(len(time) - windowW + 1)]
        sig = temp.ravel() * 27 * 1000

        newTemp = {}
        winList = []
        self.winListNames = [
            'triang',  # minimum info save
            'blackmanharris',  # middle info save (maybe best practice)
            'flattop',  # maximum info save
            'boxcar', 'blackman', 'hamming', 'hann',
            'bartlett', 'parzen', 'bohman', 'nuttall', 'barthann'
        ]

        for name in self.winListNames:
            winList.append(signal.get_window(name, windowW))

        for index, i in enumerate(winList):
            newTemp.update({
                self.winListNames[index]:
                    signal.convolve(sig, winList[index], mode='valid') / sum(winList[index])
            })

        self.time = newTime
        self.temperature = newTemp

    def get_data(self, discharge, channel, source, filterType, windowW):
        DB = DBModel.LoadDB(discharge, channel, source)

        time = DB.time
        temp = DB.temperature

        self.time_original = time.ravel()
        self.temperature_original = temp.ravel() * 27 * 1000

        self.filter_switcher(filterType, time, temp, windowW)

    def filter_switcher(self, filterType, time, temp, windowW):
        filter_switcher = {
            'custom'    : self.filtering_custom(time, temp, windowW),
            'single'    : self.filtering_single(time, temp, windowW),
            'multiple'  : self.filtering_multiple(time, temp, windowW)
        }

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

    @property
    def time_original(self):
        return self.internal_time_original

    @time_original.setter
    def time_original(self, value):
        self.internal_time_original = value

    @property
    def temperature_original(self):
        return self.internal_temperature_original

    @temperature_original.setter
    def temperature_original(self, value):
        self.internal_temperature_original = value
