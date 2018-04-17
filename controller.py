import model as DBModel
from math import fabs, floor
import numpy as np
from scipy import signal

class dataPreparing:


    def setNewTime(self, newTime):
        self.t = newTime

    def setNewTemp(self, newTemp):
        self.temperature = newTemp


    def filteringCustom(self, time, temp, windowW):

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

    def filteringSingle(self, time, temp, windowW):

        """ Window filtering with single window """
        sig = temp.ravel() * 27 * 1000
        win = signal.triang(windowW)
        newTemp = signal.convolve(sig, win, mode='valid') / sum(win)
        newTime = time[0:(len(time) - windowW + 1)]

        self.setNewTime(newTime)
        self.setNewTemp(newTemp)

    def filteringMultiple(self, time, temp, windowW):

        """ Window filtering with multiple windows """
        newTime = time[0:(len(time) - windowW + 1)]
        sig = temp.ravel() * 27 * 1000
        newTemp = {}
        winList = []
        self.winListNames = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann']
        for name in self.winListNames:
            winList.append(signal.get_window(name, windowW))

        for index, i in enumerate(winList):
            newTemp.update({self.winListNames[index]: signal.convolve(sig, winList[index], mode='valid') / sum(winList[index])})

        self.setNewTime(newTime)
        self.setNewTemp(newTemp)

    def getData(self, discharge, channel, source, filterType, windowW):
        DB = DBModel.loadDB()
        DB.loadData(discharge, channel, source)
        time = DB.t
        temp = DB.temperature

        self.filterSwitcher(filterType, time, temp, windowW)


    def filterSwitcher(self, filterType, time, temp, windowW):
        filterSwitcher = {
            'custom'    : self.filteringCustom(time, temp, windowW),
            'single'    : self.filteringSingle(time, temp, windowW),
            'multiple'  : self.filteringMultiple(time, temp, windowW)
        }
