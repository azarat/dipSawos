import model as db_model
from math import fabs, floor
import numpy as np
from operator import itemgetter

from scipy import signal
from scipy import signal as signal_processor


class DataPreparing:
    internal_time_original = 0
    internal_temperature_original = 0
    internal_time = 0
    internal_temperature = 0

    def filtering_custom(self, time, temp, window_width, **params):
        """ !!! Should be removed but I don't want XD """

        """ Custom window filtering (problem with output Y scale) """
        newTemp = []
        numOfSteps = window_width  # WRONG: window_width should be discreteStepW
        discreteStepW = int(floor(len(time) / numOfSteps))
        stepW = (max(time) - min(time)) / numOfSteps
        newTime = np.arange(min(time), max(time), stepW)

        for fixT in range(0, numOfSteps):
            sumT = 0

            for index in range(fixT * discreteStepW, (fixT * discreteStepW) + discreteStepW):
                # sumT += (temp[index]) * (1 - fabs( ((index - fixT * discreteStepW) - ((discreteStepW - 1) / 2)) / (discreteStepW / 2)))
                sumT += (temp[index]) * (1 - pow((((2 * (index - fixT * discreteStepW)) / (discreteStepW - 1)) - 1), 2))

            integral = sumT / discreteStepW

            newTemp.append(integral)

    def filtering_single(self, time, temp, window_width, window_name_val):

        """
        Window filtering with single window
        """

        sig = temp.ravel()
        win = signal.get_window(window_name_val, window_width)
        newTemp = signal.convolve(sig, win, mode='valid') / sum(win)
        newTime = time[0:(len(time) - window_width + 1)]

        self.time = newTime
        self.temperature = newTemp

    def filtering_multiple(self, time, temp, window_width, **params):

        """
        Window filtering with multiple windows
        """

        print(params)
        newTime = time[0:(len(time) - window_width + 1)]
        sig = temp.ravel()

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
            winList.append(signal.get_window(name, window_width))

        for index, i in enumerate(winList):
            newTemp.update({
                self.winListNames[index]:
                    signal.convolve(sig, winList[index], mode='valid') / sum(winList[index])
            })

        self.time = newTime
        self.temperature = newTemp

    def get_data(self, discharge, channel, source, filterType, window_width, window_name):
        DB = db_model.LoadDB(discharge, channel, source)

        time = DB.time
        temp = DB.temperature

        self.time_original = time.ravel()
        self.temperature_original = temp.ravel()

        self.filter_switcher(filterType, time, temp, window_width, window_name)

    def filter_switcher(self, filterType, time, temp, window_width, window_name):
        filter_switcher_list = {
            'custom'    : self.filtering_custom,
            'multiple'  : self.filtering_multiple,
            'single'    : self.filtering_single
        }
        filter_switcher_list.get(filterType)(time, temp, window_width, window_name_val=window_name)

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


class DataPreparingMulti(DataPreparing):
    """
    Change some logic
    """

    def __init__(self, discharge, channel, source, window_width, window_name):
        DB = db_model.LoadDB(discharge, channel, source)

        time = DB.time
        temp = DB.temperature

        self.time_original = time
        self.temperature_original = temp

        self.filter_2d(temp, window_width, window_name)

    def filter_2d(self, sig, window_width, window_name_val):

        win = signal.get_window(window_name_val, window_width)
        new_temp = {}

        for key, temp in sig.items():
            new_temp.update({
                key: signal.convolve(temp, win, mode='valid') / sum(win)
            })

        self.temperature = new_temp


class Profiling(DataPreparing):
    internal_channels_pos = 0

    """
    Change some logic
    """

    def __init__(self, discharge, channel, source):
        db = db_model.LoadDB(discharge, channel, source)

        self.channels_pos = db.channels
        self.time_original = db.time
        self.temperature_original = db.temperature

    @staticmethod
    def filter(input_signal, window_width, window_name):
        window = signal_processor.get_window(window_name, window_width)
        output_signal = {}

        for channel, temperature in input_signal.items():
            output_signal.update({
                channel: signal.convolve(temperature, window, mode='valid') / sum(window)
            })

        return output_signal

    @staticmethod
    def dict_to_list(dict_array):
        return [value for key, value in dict_array.items()]

    @staticmethod
    def list_to_dict(list_array):
        return {i: k for i, k in enumerate(list_array)}

    def order_by_r_maj(self, temperature_list_to_order, channel_from, channel_to):
        temperature_ordered_list = []

        for index, channel in enumerate(sorted(self.channels_pos.items(), key=itemgetter(1))):
            if channel[0] in range(channel_from, channel_to - 1):
                temperature_ordered_list.append(
                    temperature_list_to_order[channel[0]]
                )

        return temperature_ordered_list

    @staticmethod
    def calibrate(temperature_list_to_calibrate, public_temperature, window_width, public_window_width, channel_from, channel_to):
        calibrate_temperature_list = {}

        # print(temperature_list_to_calibrate)
        for channel in range(channel_from - 1, channel_to - 2):
            average_ece = sum(temperature_list_to_calibrate[channel][0:window_width - 1]) / window_width
            public_average_ece = sum(public_temperature[channel][0:public_window_width - 1]) / public_window_width

            calibrate_temperature_list.update({
                channel: temperature_list_to_calibrate[channel] * public_average_ece / average_ece
            })

        return calibrate_temperature_list

    @property
    def channels_pos(self):
        return self.internal_channels_pos

    @channels_pos.setter
    def channels_pos(self, value):
        self.internal_channels_pos = value

