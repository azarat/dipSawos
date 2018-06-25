import project.model as db_model
from operator import itemgetter
from scipy import signal as signal_processor
import numpy as np
import sys


class Controller:
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

        # self.channels_pos = db.channels_model
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


class Profiling(Controller):

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

    def normalization(self, temperature):
        """ -----------------------------------------
            version: 0.2
            desc: math normalization on 1
            :param temperature: 2d list of num
            :return normalized 2d list of num
        ----------------------------------------- """
        output = []

        for num_list in temperature:
            normalized = num_list / (sum(num_list[0:9]) / 10)
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
        filtered_list = []

        for num_list in temperature:
            filtered_num = []
            for num in num_list:

                if num < boundary[0]:
                    filtered_num.append(boundary[0])
                elif num > boundary[1]:
                    filtered_num.append(boundary[1])
                else:
                    filtered_num.append(num)

            filtered_list.append(filtered_num)

        return filtered_list


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


class FindInvRadius:

    @staticmethod
    def plane_indicator(plane_list, outlier):

        """ -----------------------------------------
            version: 0.2
            desc: define if list of nums (T(r_maj)) increase or decrease
            :param plane_list: 1d list of num
            :return indicator => 1: increase, -1: decrease, 0: flat
        ----------------------------------------- """

        compare_num = plane_list[0]
        increase = []
        decrease = []
        stat_weight = 1 / len(plane_list)

        for num in plane_list:

            difference = num - compare_num
            compare_num = num

            if difference > 0:
                increase.append(difference)
            elif difference < 0:
                decrease.append(difference)

        indicator_increase = abs(sum(increase) * (len(increase) * stat_weight))
        indicator_decrease = abs(sum(decrease) * (len(decrease) * stat_weight))

        # Tests
        # print(indicator_decrease)
        # print(indicator_increase)

        if indicator_increase > indicator_decrease and abs(indicator_increase - indicator_decrease) < outlier:
            indicator = 1
        elif indicator_decrease > indicator_increase and abs(indicator_decrease - indicator_increase) < outlier:
            indicator = -1
        else:
            indicator = 0

        return indicator

    def inv_radius(self, temperature_list, window_width, outlier):

        """ -----------------------------------------
            version: 0.2
            desc: define if list of nums increase or decrease
            :param temperature_list: 2d array of nums normalised on 1
            :param window_width: int val of len by which make plane indicating
            :return main_candidate_index: value of the most probable index
                    of channel with inversion radius
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)
        mean = sum(temperature_list[0]) / len(temperature_list[0])  # normalised on 1
        area = int((len(temperature_list[0]) - (len(temperature_list[0]) % window_width)))
        candidate_list = []
        stat_weight = {}

        for timeline, t_list in enumerate(temperature_list):
            candidates = []
            for i in range(window_width, area, window_width):

                analysis_area = t_list[i:i + window_width]

                # Tests
                # if i == 35:
                #     print(analysis_area)
                #     self.plane_indicator(analysis_area, outlier)

                """ Analysis only upward trends """
                plane_area_direction = self.plane_indicator(analysis_area, outlier)
                if plane_area_direction == 1:

                    """ Analysis only analysis_area which have intersection with mean value"""
                    upper_area = 0
                    under_area = 0
                    for t_analysis in analysis_area:
                        upper_area = 1 if t_analysis > mean else upper_area
                        under_area = 1 if t_analysis < mean else under_area

                    """ Candidates => (range of points, temperature at each of point) """
                    if upper_area == 1 and under_area == 1:
                        candidates.append((range(i, i + window_width), analysis_area))
                        stat_weight_to_update = (stat_weight[i] + 1) if i in stat_weight else 0
                        stat_weight.update({i: stat_weight_to_update})

            """ Candidate_list => (timeline with candidates, candidates) """
            candidate_list.append((timeline, candidates))

        search_area = max(stat_weight.items(), key=itemgetter(1))[0]

        temperature_list = np.transpose(temperature_list)
        main_candidate_index = self.sum_deviation(temperature_list[search_area:(search_area + window_width)], mean)
        main_candidate_index = search_area + main_candidate_index

        return main_candidate_index

    @staticmethod
    def sum_deviation(search_area, mean):

        """ -----------------------------------------
            version: 0.2
            desc: calculate deviation from mean val and give index of channel
            :param search_area: 2d array of nums normalised on 1 where can be channel with inv radius
            :param mean: float mean val of temperature at the very beginning after norm on 1
            :return index of channel with minimum deviation from mean that means
                    that it is index of inversion radius
        ----------------------------------------- """

        deviation = []
        for t_list in search_area:
            deviation.append(sum(abs(t_list - mean)))

        return deviation.index(min(deviation))
