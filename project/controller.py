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

    def order_by_r_maj(self, temperature_list_to_order, chan_pos_order_buffer):
        """ -----------------------------------------
            version: 0.2
            desc: ordering temperature list by r_maj position
            ;:param temperature_list_to_order: 2d array of temperature
            :return ordered 2d array
        ----------------------------------------- """
        temperature_ordered_list = []

        for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1)):
            # if channel[0] in range(1, len(temperature_list_to_order)):
            if channel[0] in temperature_list_to_order:
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
            normalized = num_list / (sum(num_list[0:10]) / 10)
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

    def collapse_duration(self, temperature_list_reverse, temperature_list, std_low_limit, inv_radius_channel, dynamic_outlier_limitation):

        """ -----------------------------------------
            version: 0.3
            desc: search time points of Collapse duration
            :param temperature_list: 2d list of num
            :param time_list: 1d list of num
            :param std_low_limit: float val of min deviation which indicate start index
            :return list with int val of indexes in time_list
        ----------------------------------------- """

        collapse_start_time = self.collapse_start(temperature_list, inv_radius_channel)
        collapse_end_time = self.collapse_end(temperature_list_reverse, inv_radius_channel, collapse_start_time)

        return (collapse_start_time, collapse_end_time)

    @staticmethod
    def collapse_end(temperature_list, inv_radius_channel, start):

        """ -----------------------------------------
            version: 0.4
            desc: search time point at which end Precursor + Fast Phase
            :param temperature_list: 2d list of num
            :return int val of index in time_list
        ----------------------------------------- """

        if inv_radius_channel == 0:
            inv_radius_channel = 60

        temperature_list = np.transpose(temperature_list[:inv_radius_channel])
        temperature_list = temperature_list[10:-10]

        # ########################################## V06 INTERESTING
        # end = 0
        # corelator = []
        # for t_list_i, t_list in enumerate(temperature_list):
        #     if t_list_i == 0:
        #         continue
        #     coef = t_list / temperature_list[t_list_i - 1]
        #     coef = np.mean(coef)  # mean or std
        #     corelator.append(coef)
        # corelator = np.array(corelator)
        # corelator = np.abs(corelator - 1)  # skip if std instead mean
        # corelator = signal_processor.medfilt(corelator, 13)[51:]
        # std = corelator
        # max_std = max(corelator[:700]) + np.std(corelator[:700]) * 10
        #
        # if max(corelator[:700]) / max(corelator) > 0.5:
        #     max_std = max(corelator[:700]) + np.std(corelator[:700]) * 1
        # elif max(corelator[:700]) / max(corelator) > 0.3:
        #     max_std = max(corelator[:700]) + np.std(corelator[:700]) * 2
        #
        # for t_i, t in enumerate(corelator):
        #     if t > max_std and end == 0:
        #             end = t_i - 10
        #
        # # print(max(corelator[:700]) / max(corelator))
        # # exit()
        # ##########################################
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 7)
        # max_std = [max_std for x in corelator]
        # ax.plot(max_std)
        # ax.plot(corelator)
        # # for t in range(0, 1900, 100):
        # #     ax.plot(temperature_list[t])
        #
        # ax.set(xlabel='Inv. Time', ylabel='Coefficient',
        #        title='Correlation coefficient')
        #
        # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #              ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(17)
        # # plt.show()
        # # exit()

        ########################################## V05 INTERESTING
        end = 0
        corelator = []
        for t_list_i, t_list in enumerate(temperature_list):
            if t_list_i < 10:
                continue
            coef = t_list / temperature_list[t_list_i - 10]
            coef = np.std(coef)
            corelator.append(coef)
        corelator = signal_processor.medfilt(corelator, 5)[6:]
        std = corelator

        if max(corelator[:700]) / max(corelator) > 0.4:
            max_std = max(corelator[:700]) + np.std(corelator[:700]) * 5
        else:
            max_std = (max(corelator) - min(corelator[:700])) / 2

        for t_i, t in enumerate(corelator):
            if t > max_std and end == 0:
                    end = t_i - 10

        ##########################################
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 7)
        max_std = [max_std for x in corelator]
        ax.plot(max_std)
        ax.plot(corelator[::-1])

        ax.set(xlabel='Inv. Time', ylabel='Coefficient',
               title='Correlation coefficient')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(17)
        # plt.show()
        # exit()

        ########################################## V04
        # end = 0
        # std = []
        # for t_i, t in enumerate(temperature_list):
        #     std.append(np.std(t))
        #
        # """
        # We know that from 0 to 700th point figure is flat
        # (first ~10 can be outliers due to median filtration)
        # """
        # std = signal_processor.medfilt(std, 51)[51:]
        # max_std = max(std[:700]) + np.std(std[:700]) * 10
        #
        # for t_i, t in enumerate(std):
        #     if t > max_std and std[t_i - 10] > max_std and end == 0:
        #         end = t_i - 20
        #
        # ##########################################
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 7)
        # max_std = [max_std for x in std]
        # ax.plot(max_std)
        # ax.plot(std)
        #
        # ax.set(xlabel='Inv. Time', ylabel='Coefficient',
        #        title='Correlation coefficient')
        #
        # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #              ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(17)
        # plt.show()
        # exit()
        #
        ########################################## V03
        #
        # std = []
        # for t in range(len(temperature_list)):
        #     std.append(np.std(temperature_list[t]))
        # std = signal_processor.medfilt(std, 51)[51:-51]
        #
        # # std = std[::-1]
        # std_norm = sum(std[:200]) / 200
        # std = std / std_norm
        #
        # level = (np.std(std[:500]) * 4) + max(std[:500])
        #
        # """ MUST TO DO. CHANGE LOGIC"""
        # end = 0
        # for l_i, l in enumerate(std):
        #     if l > level:
        #         end = l_i - 51
        #         break

        end = len(std) - end

        # colerator = []
        # cor_prev = 0
        # end = 0
        # end_up = 0
        # end_down = 10000000
        # indicator_up = 1.01
        # indicator_down = 0.99
        # for val_i, val in enumerate(std):
        #     if val_i == 0:
        #         cor_prev = val
        #     else:
        #         cor = cor_prev / val
        #         colerator.append(cor)
        #
        #         cor_prev = val
        #
        # colerator = signal_processor.medfilt(colerator, 11)[11:-11]
        # for c_i, c in enumerate(colerator):
        #     if c > indicator_up > 0 and colerator[c_i - 1] > indicator_up > 0:
        #         if end_up == 0:
        #             end_up = c_i + 10
        #
        #     if c < indicator_down > 0 and colerator[c_i - 1] < indicator_down > 0:
        #         if end_down == 10000000:
        #             end_down = c_i + 10
        #
        #     if end_down < end_up:
        #         end = end_down
        #     else:
        #         end = end_up
        #
        # end = len(std) - end

        # inds_up = [indicator_up for x in colerator]
        # inds_down = [indicator_down for x in colerator]
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 7)
        # ax.plot(inds_up)
        # ax.plot(inds_down)
        # ax.plot(colerator)
        #
        # ax.set(xlabel='Inv. Time', ylabel='Coefficient',
        #        title='Correlation coefficient')
        #
        # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #              ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(17)
        #
        # plt.show()
        # exit()

        return end

    @staticmethod
    def collapse_start(temperature_list, r_inv_index):

        """ -----------------------------------------
            version: 0.3
            desc: search time point at which start Precursor Phase
            :param temperature_list: 2d list of num
            :param std_low_limit: float val of min deviation which indicate start index
            :return int val of index in time_list
        ----------------------------------------- """

        if r_inv_index == 0:
            r_inv_index = 60

        temperature_list = np.transpose(temperature_list[:r_inv_index])
        temperature_list = temperature_list[10:-10]


        # start = 0
        # std = []
        # for t_i, t in enumerate(temperature_list):
        #     std.append(np.std(t))

        """ 
        We know that from 0 to 700th point figure is flat
        (first ~10 can be outliers due to median filtration)
        """
        # std = signal_processor.medfilt(std, 51)
        # max_std = max(std[10:700]) + np.std(std[10:700]) * 15
        # # max_std = (max_std + max(std) / 4) / 2
        #
        # for t_i, t in enumerate(std):
        #     if t > max_std and std[t_i - 10] > max_std and start == 0:
        #         start = t_i - 20

        start = 0
        corelator = []
        for t_list_i, t_list in enumerate(temperature_list):
            if t_list_i < 10:
                continue
            coef = t_list / temperature_list[t_list_i - 10]
            coef = np.std(coef)
            corelator.append(coef)
        corelator = corelator / (sum(corelator[:700]) / 700)
        corelator = signal_processor.medfilt(corelator, 5)[6:]
        std = corelator

        if max(corelator[:700]) / max(corelator) > 0.4:
            max_std = max(corelator[:700]) + np.std(corelator[:700]) * 5
        else:
            max_std = (max(corelator) - min(corelator[:700])) / 2

        for t_i, t in enumerate(corelator):
            if t > max_std and start == 0:
                start = t_i - 10

        ###############################################
        # import matplotlib.pyplot as plt
        # inds_up = [max_std for x in std]
        # fig, ax = plt.subplots()
        # fig.set_size_inches(15, 7)
        # ax.plot(inds_up)
        # ax.plot(corelator)
        #
        # ax.set(xlabel='Inv. Time', ylabel='Coefficient',
        #        title='Correlation coefficient')
        #
        # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #              ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(17)
        #
        # # plt.show()
        # # exit()
        ###############################################

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

    def inv_radius(self, temperature_list, window_width, std_low_limit, channel_offset):

        """ -----------------------------------------
            version: 0.2
            desc: define if list of nums increase or decrease
            :param temperature_list: 2d array of nums normalised on 1
            :param std_low_limit: float val to skip flat regions
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
            if timeline == 0:
                continue

            # flat_outlier = sum(abs(t_list - mean)) / len(t_list)
            flat_outlier = np.std(t_list)

            # print(flat_outlier, ' ', std_low_limit)
            if flat_outlier > std_low_limit:
                candidates = []
                plane_area_direction_prev = 1
                for i in range(window_width, area, window_width):
                    if i < channel_offset:
                        continue

                    analysis_area = t_list[i-1:i + window_width]

                    """ Analysis only upward trends """
                    plane_area_direction = self.trend_indicator(analysis_area)
                    if plane_area_direction != -1 and plane_area_direction_prev == 1:

                        """ Analysis only analysis_area which have intersection with mean value"""
                        upper_area = 0
                        under_area = 0
                        for t_i, t_analysis in enumerate(analysis_area):
                            upper_area = 1 if t_analysis > mean else upper_area
                            under_area = 1 if t_analysis < mean else under_area

                        """ Candidates => (range of points, temperature at each point) """
                        if upper_area == 1 and under_area == 1:
                            candidates.append((range(i, i + window_width), analysis_area))
                            stat_weight_to_update = (stat_weight[i] + 1) if i in stat_weight else 0
                            stat_weight.update({i: stat_weight_to_update})

                    plane_area_direction_prev = plane_area_direction

                    """ Candidate_list => (timeline with candidates, candidates) """
                    candidate_list.append((timeline, candidates))

        if len(stat_weight) == 0:
            # print('Error: std_low_limit is too big')
            return 0

        search_area_max = max(stat_weight.items(), key=itemgetter(1))
        search_area = search_area_max[0]

        # candidate_info = 0
        # if search_area_max[1] < 1000:
        #     return candidate_info

        temperature_list = np.transpose(temperature_list)
        main_candidate_index = self.sum_deviation(temperature_list[search_area:(search_area + window_width)])
        main_candidate_index = search_area + main_candidate_index

        return main_candidate_index

    def inv_radius_intersection(self, temperature_list, window_width, std_low_limit, r_maj):

        """ -----------------------------------------
            version: 0.2.1
            desc: define if list of nums increase or decrease
            :param temperature_list: 2d array of nums normalised on 1
            :param std_low_limit: float val to skip flat regions
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

            if flat_outlier > std_low_limit:
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
            sys.exit('Error: std_low_limit is too big')

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
    def sum_deviation(search_area):

        """ -----------------------------------------
            version: 0.3
            desc: calculate deviation from mean val and give index of channel
            :param search_area: 2d array of nums normalised on 1 where can be channel with inv radius
            :return index of channel with minimum deviation from mean that means
                    that it is index of inversion radius
        ----------------------------------------- """

        deviation = []
        for t_list in search_area:
            deviation.append(np.std(t_list))

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
            'ece_data': mat['ece_data'][0, 0]['signal'],
            'discharge': mat['ece_data'][0, 0]['discharge']
        }

        return data

    @staticmethod
    def ml_find_inv_radius(data_train, data_test):
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        data_test = np.array(data_test)[1000:1900, :60]

        print('data_train len:', len(data_train))

        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]

        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()
        model = DecisionTreeClassifier().fit(X_train, y_train)

        print('Accuracy of Linear SVC classifier on training set: {:.2f}'
         .format(model.score(X_train, y_train)))

        return model.predict(data_test)

    @staticmethod
    def ml_find_collapse_duration(data_train, data_test):
        from sklearn.tree import DecisionTreeClassifier

        data_test = np.array(data_test)[15:45, 1000:1901]

        print('data_train len:', len(data_train))
        # print(data_train.shape)
        # exit()

        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]
        # print(y_train[0])
        # exit()

        model = DecisionTreeClassifier().fit(X_train, y_train)

        print('Accuracy of Linear SVC classifier on training set: {:.2f}'
              .format(model.score(X_train, y_train)))

        # exit()
        return model.predict(data_test)