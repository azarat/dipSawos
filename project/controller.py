import project.model as db_model
from operator import itemgetter
from scipy import signal as signal_processor
import numpy as np


class DataPreparing:
    internal_time_original = 0
    internal_temperature_original = 0
    internal_channels_pos = 0

    def __init__(self, discharge, channel, source):
        db = db_model.LoadDB(discharge, channel, source)

        self.win_list_names = [
            'triang',  # minimum info save
            'blackmanharris',  # middle info save (maybe best practice)
            'flattop',  # maximum info save
            'boxcar', 'blackman', 'hamming', 'hann',
            'bartlett', 'parzen', 'bohman', 'nuttall', 'barthann'
        ]

        self.channels_pos = db.channels
        self.time_original = db.time
        self.temperature_original = db.temperature

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

    @property
    def channels_pos(self):
        return self.internal_channels_pos

    @channels_pos.setter
    def channels_pos(self, value):
        self.internal_channels_pos = value


class Profiling(DataPreparing):

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

        for channel in range(channel_from - 1, channel_to - 2):
            average_ece = sum(temperature_list_to_calibrate[channel][0:window_width - 1]) / window_width
            public_average_ece = sum(public_temperature[channel][0:public_window_width - 1]) / public_window_width

            calibrate_temperature_list.update({
                channel: temperature_list_to_calibrate[channel] * public_average_ece / average_ece
            })

        return calibrate_temperature_list


class PreProfiling:

    @staticmethod
    def filter(input_signal, window_width, window_name):
        window = signal_processor.get_window(window_name, window_width)
        output_signal = {}

        for channel, temperature in input_signal.items():
            output_signal.update({
                channel: signal_processor.convolve(temperature, window, mode='valid') / sum(window)
            })

        return output_signal

    @staticmethod
    def dict_to_list(dict_array):
        return [value for key, value in dict_array.items()]

    @staticmethod
    def list_to_dict(list_array):
        return {i: k for i, k in enumerate(list_array)}

    def calculate_deviation(self, window_width, temperature_original, analyze_window, channel_to_compare):
        rmsd = []

        for index, name in enumerate(analyze_window):

            """ Prepare data for calculate """
            filtered_temperature = self.list_to_dict(temperature_original)
            filtered_temperature = self.filter(filtered_temperature, window_width, name)
            filtered_temperature = self.dict_to_list(filtered_temperature)[channel_to_compare]

            filtered_temperature_size = len(filtered_temperature)
            cut_temperature_original = temperature_original[channel_to_compare][0:filtered_temperature_size]

            """ Calculate RMSD (Root-Mean-Square Deviation) """
            rmsd_array = pow(filtered_temperature - cut_temperature_original, 2)
            rmsd.append(
                np.sqrt(np.sum(rmsd_array) / len(rmsd_array))
            )

        return rmsd

