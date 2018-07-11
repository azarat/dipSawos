import project.model as db_model
from operator import itemgetter
from scipy import signal as signal_processor
import numpy as np
import sys


class Controller:
    internal_time_original = 0
    internal_temperature_original = 0
    internal_channels_pos = 0

    win_list_names = [
        'triang',  # minimum info save
        'blackmanharris',  # middle info save (maybe best practice)
        'flattop',  # maximum info save
        'boxcar', 'blackman', 'hamming', 'hann',
        'bartlett', 'parzen', 'bohman', 'nuttall', 'barthann'
    ]

    def load(self, discharge, source):
        db = db_model.LoadDB(discharge, source)

        # self.channels_pos = db.channels_model
        self.channels_pos = db.channels
        self.time_original = db.time
        self.temperature_original = db.temperature

        return 1

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
    def normalization(temperature):

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
            version: 0.2.0
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

    @staticmethod
    def outlier_filter_std_deviation(temperature_list, boundary, offset):

        """ -----------------------------------------
            version: 0.2.1
            desc: remove extra values from list
            :param temperature_list: 1d list of num
            :param boundary: float val intensity of cutting edge on standard deviation
            :param offset: int val of which temperature val reset
            :return filtered 1d list of num
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)

        filtered_list = []

        for i_list, temperature in enumerate(temperature_list):
            filtered_num = []

            mean, data_std = np.mean(temperature), np.std(temperature)
            cut_off = data_std * boundary
            lower, upper = mean - cut_off, mean + cut_off

            for i, t in enumerate(temperature):
                if t < lower:
                    filtered_num.append(temperature[i-offset])
                    # filtered_num.append(lower)
                elif t > upper:
                    filtered_num.append(temperature[i-offset])
                    # filtered_num.append(upper)
                else:
                    filtered_num.append(t)

            filtered_list.append(filtered_num)

        return np.transpose(filtered_list)


class PreProfiling:

    @staticmethod
    def median_filtered(signal, threshold=3):
        """
        signal: is numpy array-like
        returns: signal, numpy array
        """
        difference = np.abs(signal - np.median(signal))
        median_difference = np.median(difference)
        s = 0 if median_difference == 0 else difference / float(median_difference)
        mask = s > threshold
        signal[mask] = np.median(signal)
        return signal

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


class FindCollapseDuration:

    def collapse_duration(self, temperature_list, flat_outlier_limitation, inv_radius_channel, std_multiplicity):

        """ -----------------------------------------
            version: 0.2
            desc: search time points of Collapse duration
            :param temperature_list: 2d list of num
            :param time_list: 1d list of num
            :param flat_outlier_limitation: float val of min deviation which indicate start index
            :return list with int val of indexes in time_list
        ----------------------------------------- """

        collapse_start_time = self.collapse_start(temperature_list, flat_outlier_limitation)
        collapse_end_time = self.collapse_end(temperature_list, inv_radius_channel, std_multiplicity)

        return (collapse_start_time, collapse_end_time)

    @staticmethod
    def collapse_end(temperature_list, inv_radius_channel, std_multiplicity):

        """ -----------------------------------------
            version: 0.2
            desc: search time point at which end Precursor + Fast Phase
            :param temperature_list: 2d list of num
            :return int val of index in time_list
        ----------------------------------------- """

        collapse_list = []

        for channel, temperature in enumerate(temperature_list):

            if channel > inv_radius_channel:
                continue

            collapse_end_time = 0

            temperature_rev = temperature[::-1]
            for i, t in enumerate(temperature_rev):

                if i < 10:
                    continue

                analysis_area = temperature_rev[0:i]

                mean, data_std = np.mean(analysis_area), np.std(analysis_area)
                cut_off = data_std * std_multiplicity
                lower, upper = mean - cut_off, mean + cut_off

                if t < lower or t > upper:
                    collapse_end_time = i
                    break

            if collapse_end_time > 0:
                collapse_list.append(collapse_end_time)

        return len(temperature_list[0]) - min(collapse_list)

    @staticmethod
    def collapse_start(temperature_list, flat_outlier_limitation):

        """ -----------------------------------------
            version: 0.2
            desc: search time point at which start Precursor Phase
            :param temperature_list: 2d list of num
            :param flat_outlier_limitation: float val of min deviation which indicate start index
            :return int val of index in time_list
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)
        mean = sum(temperature_list[0]) / len(temperature_list[0])
        start = 0

        for timeline, t_list in enumerate(temperature_list):
            flat_outlier = sum(abs(t_list - mean)) / len(t_list)

            if flat_outlier > flat_outlier_limitation:
                start = timeline
                break

        return start


class FindInvRadius:

    @staticmethod
    def trend_indicator(plane_list):

        """ -----------------------------------------
            version: 0.2
            desc: define if list of nums (T(r_maj)) increase or decrease
            :param plane_list: 1d list of num
            :return indicator => 1: increase, -1: decrease, 0: flat/undefined
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

        indicator_increase = sum(map(abs, increase)) * (len(increase) * stat_weight) + 1
        indicator_decrease = sum(map(abs, decrease)) * (len(decrease) * stat_weight) + 1

        if indicator_increase > indicator_decrease:
            indicator = 1
        elif indicator_decrease > indicator_increase:
            indicator = -1
        else:
            indicator = 0

        return indicator

    def inv_radius(self, temperature_list, window_width, flat_outlier_limitation):

        """ -----------------------------------------
            version: 0.2
            desc: define if list of nums increase or decrease
            :param temperature_list: 2d array of nums normalised on 1
            :param flat_outlier_limitation: float val to skip flat regions
            :param window_width: int val of len by which make plane indicating
            :return main_candidate_index: value of the most probable index
                    of channel with inversion radius
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)
        mean = np.mean(temperature_list[0])
        area = int((len(temperature_list[0]) - (len(temperature_list[0]) % window_width)))
        candidate_list = []
        stat_weight = {}

        for timeline, t_list in enumerate(temperature_list):
            flat_outlier = sum(abs(t_list - mean)) / len(t_list)

            # print(flat_outlier, ' ', flat_outlier_limitation)
            if flat_outlier > flat_outlier_limitation:
                candidates = []
                plane_area_direction_prev = 1
                for i in range(window_width, area, window_width):

                    analysis_area = t_list[i-1:i + window_width]

                    """ Analysis only upward trends """
                    plane_area_direction = self.trend_indicator(analysis_area)
                    if plane_area_direction != -1 and plane_area_direction_prev == 1:

                        """ Analysis only analysis_area which have intersection with mean value"""
                        upper_area = 0
                        under_area = 0
                        for t_analysis in analysis_area:
                            upper_area = 1 if t_analysis > mean else upper_area
                            under_area = 1 if t_analysis < mean else under_area
                            # print(under_area, ' ', upper_area)

                        """ Candidates => (range of points, temperature at each point) """
                        if upper_area == 1 and under_area == 1:
                            candidates.append((range(i, i + window_width), analysis_area))
                            stat_weight_to_update = (stat_weight[i] + 1) if i in stat_weight else 0
                            stat_weight.update({i: stat_weight_to_update})

                    plane_area_direction_prev = plane_area_direction

                    """ Candidate_list => (timeline with candidates, candidates) """
                    candidate_list.append((timeline, candidates))

        if len(stat_weight) == 0:
            sys.exit('Error: flat_outlier_limitation is too big')

        search_area_max = max(stat_weight.items(), key=itemgetter(1))
        search_area = search_area_max[0]

        candidate_info = 0
        if search_area_max[1] < 1000:
            return candidate_info

        temperature_list = np.transpose(temperature_list)
        main_candidate_index = self.sum_deviation(temperature_list[search_area:(search_area + window_width)], mean)
        main_candidate_index = search_area + main_candidate_index

        return main_candidate_index

    def inv_radius_intersection(self, temperature_list, window_width, flat_outlier_limitation, r_maj):

        """ -----------------------------------------
            version: 0.2.1
            desc: define if list of nums increase or decrease
            :param temperature_list: 2d array of nums normalised on 1
            :param flat_outlier_limitation: float val to skip flat regions
            :param window_width: int val of len by which make plane indicating
            :return main_candidate_index: value of the most probable index
                    of channel with inversion radius
        ----------------------------------------- """

        temperature_list = np.transpose(temperature_list)
        mean = sum(temperature_list[0]) / len(temperature_list[0])  # normalised on 1
        area = int((len(temperature_list[0]) - (len(temperature_list[0]) % window_width)))
        stat_weight = {}
        candidates = []

        for timeline, t_list in enumerate(temperature_list):
            flat_outlier = sum(abs(t_list - mean)) / len(t_list)

            if flat_outlier > flat_outlier_limitation:
                plane_area_direction_prev = 1
                for i in range(window_width, area, window_width):

                    analysis_area = t_list[i-1:i + window_width]

                    """ Analysis only upward trends """
                    plane_area_direction = self.trend_indicator(analysis_area)
                    if plane_area_direction != -1 and plane_area_direction_prev == 1:

                        """ Analysis only analysis_area which have intersection with mean value"""
                        upper_area = 0
                        under_area = 0
                        intersection_indexes = ()
                        for tia, t_analysis in enumerate(analysis_area):
                            upper_area = 1 if t_analysis > mean else upper_area
                            if under_area == 1 and upper_area == 1:
                                intersection_indexes = (i + tia - 2, i + tia - 1)
                                break
                            under_area = 1 if t_analysis < mean else under_area

                        """ Candidates => (range of points, temperature at each of point) """
                        if upper_area == 1 and under_area == 1 and len(intersection_indexes) == 2:

                            intersection = self.intersection_pos(t_list, intersection_indexes, r_maj, mean)
                            if intersection > 0:
                                candidates.append((intersection_indexes[0], intersection))

                                stat_weight_to_update = (stat_weight[intersection_indexes[0]] + 1) if intersection_indexes[0] in stat_weight else 1
                                stat_weight.update({intersection_indexes[0]: stat_weight_to_update})

                    plane_area_direction_prev = plane_area_direction

        sorted_candidates = {}
        # print(len(candidates))
        for c in candidates:
            inter_list = sorted_candidates[c[0]] if c[0] in sorted_candidates else []
            inter_list.append(c[1])
            sorted_candidates.update({c[0]: inter_list})

        if len(stat_weight) == 0:
            sys.exit('Error: flat_outlier_limitation is too big')

        search_area_max = max(stat_weight.items(), key=itemgetter(1))
        search_area = search_area_max[0]

        intersection = np.mean(sorted_candidates[search_area])

        candidate_info = (0, 0)
        if search_area_max[1] < 500:
            return candidate_info

        candidate_info = ('{:.4f}'.format(intersection), str(search_area) + "-" + str(search_area + 1))

        return candidate_info

    @staticmethod
    def intersection_pos(t_list, intersection_indexes, r_maj, mean):

        """ -----------------------------------------
            version: 0.2
            desc: calculate deviation from mean val and give index of channel
            :param t_list:
            :param intersection_indexes:
            :param r_maj:
            :param mean:
            :return intersection float val
        ----------------------------------------- """

        cathete_under = mean - t_list[intersection_indexes[0]]
        cathete_above = t_list[intersection_indexes[1]] - mean

        cathete_side = cathete_under + cathete_above

        if cathete_side > 0:
            cathete_down = r_maj[intersection_indexes[1]] - r_maj[intersection_indexes[0]]

            hypotenuse = np.sqrt(np.power(cathete_side, 2) + np.power(cathete_down, 2))

            aspect_ratio = hypotenuse / cathete_side
            hypotenuse_under = aspect_ratio * cathete_under

            cathete_mean = np.sqrt(np.power(hypotenuse_under, 2) - np.power(cathete_under, 2))

            intersection = r_maj[intersection_indexes[0]] + cathete_mean
        else:
            intersection = 0

        return intersection

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


class MachineLearning:

    @staticmethod
    def ml_load(filename):
        """ -----------------------------------------
             version: 0.3
             desc: load data from matlab with previously prepared data
             :param filename: string val
             :return nd array
         ----------------------------------------- """
        db = db_model.Model()
        mat = db.load(filename)

        data = {
            'ece_data': mat['ece_data'][0, 0]['signal'][0, :],
            'discharge': mat['ece_data'][0, 0]['discharge'][0, :]
        }

        return data

    def ml_find_outliers(self):
        import pandas as pd
        import matplotlib.pyplot as plt

        dataset = pd.read_csv('out.csv').values
        dataset_test = pd.read_csv('out_test.csv').values

        from sklearn.svm import SVC

        X_train = (dataset[40:, 1500:1550])
        y_train = []
        for i in range(9):
            y_train.append(0)
        y_train.append(1)

        plt.plot(dataset_test[:, 1500])
        plt.show()

        X_test = (dataset_test[:, 1600:1650])
        # print(len(X_test))
        # exit()
        y_test = []
        for i in range(49):
            y_test.append(0)
        y_test.append(1)

        this_C = 1.0
        clf = SVC(kernel='linear', C=this_C).fit(X_train, y_train)
        # print('XLF Lag1 dataset')
        # print('Accuracy of Linear SVC classifier on training set: {:.2f}'
        #  .format(clf.score(X_train, y_train)))
        # print('Accuracy of Linear SVC classifier on test set: {:.2f}'
        #  .format(clf.score(X_test, y_test)))

        return clf.predict(X_test)

    def ml_find_inv_radius(self):
        pass

    def ml_find_collapse_duration(self):
        pass
