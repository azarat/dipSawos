import project.model as db_model
from operator import itemgetter
from scipy import signal as signal_processor
import numpy as np
import sys


class DataPreparing:
    internal_time_original = 0
    internal_temperature_original = 0
    internal_channels_pos = 0

    def __init__(self, discharge, source):
        db = db_model.LoadDB(discharge, source)

        self.win_list_names = [
            'triang',  # minimum info save
            'blackmanharris',  # middle info save (maybe best practice)
            'flattop',  # maximum info save
            'boxcar', 'blackman', 'hamming', 'hann',
            'bartlett', 'parzen', 'bohman', 'nuttall', 'barthann'
        ]

        self.channels_pos = db.channels_model
        # self.channels_pos = db.channels
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

    def order_by_r_maj(self, temperature_list_to_order):
        """ -----------------------------------------
            version: 0.2
            desc: ordering temperature list by r_maj position
            ;:param temperature_list_to_order: 2d array of temperature
            :return ordered 2d array
        ----------------------------------------- """
        temperature_ordered_list = []

        for channel in sorted(self.channels_pos.items(), key=itemgetter(1)):
            if channel[0] in range(1, len(temperature_list_to_order)):
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

    def normalization(self, temperature, method):
        """ -----------------------------------------
            version: 0.2
            desc: math normalization on 1
            :param temperature: 2d list of num
            :param method: norm by "end" or "start"
            :return normalized 2d list of num
        ----------------------------------------- """
        output = []

        for num_list in temperature:
            if method == "start":
                normalized = num_list / (sum(num_list[0:9]) / 10)
                boundary = (0, 1.2)
            elif method == "end":
                normalized = num_list / (sum(num_list[len(num_list)-10:len(num_list)-1]) / 10)
                boundary = (0.8, 2)
            else:
                sys.exit("Error: normalization failed")

            normalized = self.outlier_filter(normalized, boundary)

            output.append(normalized)

        return output

    @staticmethod
    def outlier_filter(temperature, boundary):
        """ -----------------------------------------
            version: 0.2
            desc: remove extra values from list
            :param temperature: 1d list of num
            :param boundary: array 0=>min and 1=>max
            :return filtered 1d list of num
        ----------------------------------------- """
        filtered = []

        for num in temperature:
            if num < boundary[0]:
                filtered.append(boundary[0])
            elif num > boundary[1]:
                filtered.append(boundary[1])
            else:
                filtered.append(num)

        return filtered


class PreProfiling:

    @staticmethod
    def filter(input_signal, window_width, window_name):
        window = signal_processor.get_window(window_name, window_width)
        output_signal = []

        for temperature in input_signal:
            output_signal.append(
                signal_processor.convolve(temperature, window, mode='valid') / sum(window)
            )

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

