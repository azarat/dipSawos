import project.controller as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from operator import itemgetter
import os
from scipy import signal as signal_processor
import matplotlib.patches as patches
import pandas as pd

# IMPORTANT to be declared for 3d plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import scipy.io as io
import time


class ViewData:
    processing = dt.PreProfiling()
    channel_from = 0
    channel_to = 92
    window_width_val_dur = 3  # time smoothing
    window_width_val_inv = 500  # time smoothing
    window_width_rad_val = 5  # radius smoothing
    window_func = 'triang'
    boundary = (0.75, 1.5)
    # boundary = (-100, 100)

    def __init__(self):
        print("VERSION: 0.4")

        # Single dis
        dis_end = 2
        dis_start = dis_end - 1

        # Range of dis
        # dis_start, dis_end = 0, 64

        close_plots = 0

        """ Inversion radius """
        # self.build_plots_to_find_inversion_radius(start=dis_start, end=dis_end, median_filter_window_size=(3, 3),
        #                                           highlight_r_inv=1, start_offset=1, close_plots=close_plots)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Collapse duration """
        self.build_plots_to_find_collapse_time_duration(start=dis_start, end=dis_end, median_filter_window_size=(11, 11),
                                                        highlight_r_inv=1, start_offset=1, close_plots=close_plots)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Colormap overview """
        # self.build_plots_colormap(start=dis_start, end=dis_end, median_filter_window_size=(11, 11),
        #                           start_offset=4, end_offset=-30, close_plots=close_plots)
        # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        plt.show()
        # # # # # # # # # # # # # # # # # # # # # # # #

        # self.extra_normalization()
        # self.export_inversion_radius()
        # self.ml_find_inv_radius()
        # self.ml_find_collapse_duration()

    def ml_find_collapse_duration(self):
        """ -----------------------------------------
            version: 0.3
            desc: export collapse_duration to train neural network
                  save to mat files
                  every channel have sum of time points
            :return 1
        ----------------------------------------- """

        """ Extract data from MATLAB database """
        controller = dt.MachineLearning()
        data = controller.ml_load('machine_learning/train/train_ECE_data_ColDur.mat')
        data_train = data['ece_data']
        data_test = controller.ml_load('machine_learning/ECE_data_tS500_n1_mF5_uDR.mat')

        dis_count = len(data['ece_data'])  # 31
        list_dis_all = data['discharge'][0, :]
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Find collapse duration """
        data_train_union = list()
        for dis_i, dis in enumerate(data_train):
            if dis_i == 25:
                continue

            for ch_i, ch in enumerate(dis):
                if ch_i > 45 or ch_i < 15:
                    continue

                std_cleared = list()
                for num_i, num in enumerate(ch[0, :]):
                    if num_i < 1000 or num_i > 1900:
                        continue
                    std_cleared.append(num)
                std_cleared.append(ch[0, -1])

                std_cleared = np.array(std_cleared)
                data_train_union.append(std_cleared)

        data_train_union = np.array(data_train_union)
        test_dis = 11
        data_test = np.transpose(data_test['ece_data'][0, :][test_dis])

        collapse_duration_time = np.unique(controller.ml_find_collapse_duration(data_train_union, data_test))

        print(collapse_duration_time)

        """ Find Inversion Radius """
        """ Method 1 - identify closest channel """
        # # # # # # # # # # # # # # # # # # # # # # # #
        inv_radius_channel_index = dt.FindInvRadius().inv_radius(data_test, 5, 0.01)
        print("Inversion radius M1: " + str(inv_radius_channel_index))
        # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        collapse_duration_time = dt.FindCollapseDuration().collapse_duration(data_test, 6,
                                                                             inv_radius_channel_index, 1.03)
        collapse_duration_time = [int(x + (self.window_width_val_inv * 0.7)) for x in collapse_duration_time]
        print("Time segment: ", collapse_duration_time[0], " ", collapse_duration_time[1])
        print("Time duration: ", (collapse_duration_time[1] - collapse_duration_time[0]),
              " ms")
        # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        # self.build_temperature_rmaj_single_plot(
        #     data_test[:70], self.window_func, self.window_width_val_inv, range(len(data_test[0])),
        #     time_limits=collapse_duration_time, discharge=test_dis)
        # # # # # # # # # # # # # # # # # # # # # # # #

        plt.show()

        return 1

    @staticmethod
    def ml_find_inv_radius():
        """ -----------------------------------------
            version: 0.3
            desc: train 'machine learning' to find inv radius
                  based on previously prepared data with inv rad channel indicator 0/1
            :return 1
        ----------------------------------------- """

        """ Extract data from MATLAB database """
        controller = dt.MachineLearning()
        data_train = controller.ml_load('machine_learning/train/train_ECE_data_Rinv.mat')
        data_test = controller.ml_load('machine_learning/ECE_data_tS500_n1_mF5_uDR.mat')

        dis_list = data_train['discharge'][0]
        data_train = data_train['ece_data'][0]

        data_train_union = []
        for dis_i, dis in enumerate(data_train):
            if dis_i == 25:
                continue
            for ch_i, ch in enumerate(dis):
                if ch_i < 1000:
                    continue
                std_cleared = []
                for num_i, num in enumerate(ch):
                    if num_i > 59:
                        continue
                    std_cleared.append(num)

                std_cleared.append(ch[-1])
                data_train_union.append(std_cleared)

        data_train_union = np.array(data_train_union)
        test_dis = 10
        data_test = data_test['ece_data'][0, :][test_dis]

        ch_std = []
        for ch in data_train_union:
            ch_std.append(np.std(ch))

        # plt.plot(ch_std)
        # plt.show()
        # exit()

        print('Test data: ', dis_list[test_dis], ' discharge')

        # # # # # # # # # # # # # # # # # # # # # # # #

        r_inv = np.unique(controller.ml_find_inv_radius(data_train_union, data_test))

        print('------\nR_inv set:')
        print(r_inv)
        print('------')

        if len(r_inv) > 0:
            for i, val in enumerate(r_inv):
                print(i, '.', ' ', val, ' channel = R_inv')
        else:
            print('R_inv not found')

        fig, ax = plt.subplots(1, 1)
        for t in range(0, 1900, 100):
            ax.plot(data_test[t])

        # ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.show()

        return 1

    def build_plots_colormap(self, start, end, median_filter_window_size, start_offset, end_offset, close_plots):
        """ -----------------------------------------
            version: 0.3
            desc: build colormap plots
            :return 1
        ----------------------------------------- """

        print('Start detection: inversion radius')

        for dis in range(start, end):

            print("Discharge: " + str(dis + 1))

            if dis == 31:
                continue

            """ Extract data from MATLAB database """
            print('Load data')
            data = dt.Profiling()
            data.load(discharge=dis, source='real')
            # data_public = dt.Profiling(discharge=55, source='public')
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Remove all channels with R_maj = nan """
            print('Remove channels with R_maj = nan')
            r_maj_list = self.processing.dict_to_list(data.channels_pos)
            temperature_list_original = data.temperature_original

            r_maj_list_buffer = []
            temperature_list_buffer = []
            chan_order_buffer = []
            chan_pos_order_buffer = {}
            for t_list_i, t_list in enumerate(temperature_list_original.items()):

                if r_maj_list[t_list_i] > 0:
                    temperature_list_buffer.append(t_list[1])
                    chan_order_buffer.append(t_list[0])
                    chan_pos_order_buffer[t_list[0]] = r_maj_list[t_list_i]
                    r_maj_list_buffer.append(r_maj_list[t_list_i])

            temperature_list_original = temperature_list_buffer
            # print("--------------------")
            # print(chan_pos_order_buffer)
            # print("--------------------")
            r_maj_list = sorted(r_maj_list_buffer)
            temperature_list_original = {chan_order_buffer[i]: k for i, k in enumerate(temperature_list_original)}
            # # # # # # # # # # # # # # # # # # # # # # # #

            print('Sort channels by their own R_maj value')
            """ Sort R_maj index and time lists by R_maj value """
            r_maj_list_indexes = [channel[0] for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1))]
            r_maj_list = [channel[1] for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1))]
            channel_order_list = {index: channel[0] for index, channel in enumerate(sorted(chan_pos_order_buffer.items(), key=itemgetter(1)))}
            time_list = data.time_original
            """ Same vars in a.u. units """
            """ Ordering by R_maj """
            temperature_list_original = data.order_by_r_maj(temperature_list_original, chan_pos_order_buffer)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
            Filtering T(t), i.e., smoothing 
            WARNING: do not smooth due to info losses in 3d 
            """
            print('Smoothing channels along timeline - SKIP')
            # temperature_list_original = self.processing.filter(
            #     temperature_list_original, self.window_width_val_inv, self.window_func)
            temperature_list_original = temperature_list_original  # skip filtration
            # # # # # # # # # # # # # # # # # # # # # # # #


            """ Calibrate (Normalization on 1) """
            print('Normalizing channels on 1')
            temperature_list_original = data.normalization(temperature_list_original)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Median filtering IMPORTANT to remove outliers T(r_maj) """
            if median_filter_window_size != 0:
                median_filter = str(median_filter_window_size[0]) + 'x' + str(median_filter_window_size[1])
                temperature_list_original = signal_processor.medfilt2d(temperature_list_original, median_filter_window_size)
            else:
                median_filter = 0
                temperature_list_original = data.outlier_filter(temperature_list_original, self.boundary)
                temperature_list_original = np.array(temperature_list_original)

            print("--------------------")

            print('Plotting results and save as images .PNG')
            self.build_temperature_rmaj_time_3d_surface(temperature_list_original[start_offset:end_offset, start_offset:-start_offset],
                                                        r_maj_list[start_offset:end_offset],
                                                        time_list[start_offset:-start_offset],
                                                        discharge=dis, median_filter=median_filter)

            print("--------------------")
            print('\n')
            if close_plots == 1:
                plt.close("all")

        return 1

    def build_plots_to_find_collapse_time_duration(self, start, end, median_filter_window_size, highlight_r_inv, start_offset, close_plots):
        """ -----------------------------------------
            version: 0.3
            desc: build plots with collapse duration time-points
            :return 1
        ----------------------------------------- """

        print('Start detection: collapse duration')

        for dis in range(start, end):

            print("Discharge: " + str(dis + 1))

            if dis == 31:
                continue

            """ Extract data from MATLAB database """
            print('Load data')
            data = dt.Profiling()
            data.load(discharge=dis, source='real')
            # data_public = dt.Profiling(discharge=55, source='public')
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Remove all channels with R_maj = nan """
            print('Remove channels with R_maj = nan')
            r_maj_list = self.processing.dict_to_list(data.channels_pos)
            temperature_list_original = data.temperature_original

            r_maj_list_buffer = []
            temperature_list_buffer = []
            chan_order_buffer = []
            chan_pos_order_buffer = {}
            for t_list_i, t_list in enumerate(temperature_list_original.items()):

                if r_maj_list[t_list_i] > 0:
                    temperature_list_buffer.append(t_list[1])
                    chan_order_buffer.append(t_list[0])
                    chan_pos_order_buffer[t_list[0]] = r_maj_list[t_list_i]
                    r_maj_list_buffer.append(r_maj_list[t_list_i])

            temperature_list_original = temperature_list_buffer
            # print("--------------------")
            # print(chan_pos_order_buffer)
            # print("--------------------")
            r_maj_list = sorted(r_maj_list_buffer)
            temperature_list_original = {chan_order_buffer[i]: k for i, k in enumerate(temperature_list_original)}
            # # # # # # # # # # # # # # # # # # # # # # # #

            print('Sort channels by their own R_maj value')
            """ Sort R_maj index and time lists by R_maj value """
            r_maj_list_indexes = [channel[0] for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1))]
            r_maj_list = [channel[1] for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1))]
            channel_order_list = {index: channel[0] for index, channel in enumerate(sorted(chan_pos_order_buffer.items(), key=itemgetter(1)))}
            time_list = data.time_original
            """ Same vars in a.u. units """
            """ Ordering by R_maj """
            temperature_list_original = data.order_by_r_maj(temperature_list_original, chan_pos_order_buffer)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
            Filtering T(t), i.e., smoothing 
            WARNING: do not smooth due to info losses in 3d 
            """
            print('Smoothing channels along timeline')
            temperature_list_original = self.processing.filter(
                temperature_list_original, self.window_width_val_dur, self.window_func)
            # temperature_list_original = temperature_list_original  # skip filtration
            # # # # # # # # # # # # # # # # # # # # # # # #


            """ Calibrate (Normalization on 1) """
            print('Normalizing channels on 1')
            temperature_list_original = np.array(temperature_list_original)
            temperature_list_reverse = data.normalization(temperature_list_original[:, ::-1])
            temperature_list_original = data.normalization(temperature_list_original)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Median filtering IMPORTANT to remove outliers T(r_maj) """
            if median_filter_window_size != 0:
                median_filter = str(median_filter_window_size[0]) + 'x' + str(median_filter_window_size[1])
                temperature_list_original = signal_processor.medfilt2d(temperature_list_original, median_filter_window_size)
                temperature_list_reverse = signal_processor.medfilt2d(temperature_list_reverse, median_filter_window_size)
            else:
                median_filter = 0

            """ 60 - due to the low accuracy after r_inv, which have influence on r_inv detection """
            print('Inversion radius detection')
            temperature_list_rad = temperature_list_original[:60, median_filter_window_size[0]:-median_filter_window_size[0]]
            r_maj_list = r_maj_list[:len(temperature_list_rad)]
            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list=temperature_list_rad,
                                                               window_width=6, std_low_limit=0.01,
                                                               channel_offset=15)

            """ 
            Smooth T(r_maj)
            IMPORTANT to remove sawtooth behavior T(r_maj)
            """
            print('Smoothing channels along radius')
            # temperature_list_original = self.processing.filter(
            #     np.transpose(temperature_list_original), self.window_width_rad_val, self.window_func)
            # temperature_list_original = np.transpose(temperature_list_original)
            # # # # # # # # # # # # # # # # # # # # # # #

            print("--------------------")

            """ Identifying collapse duration """
            print('Determination of collapse duration')
            collapse_duration_time = dt.FindCollapseDuration().collapse_duration(temperature_list_reverse, temperature_list_original, 6, inv_radius_channel, 1.03)
            collapse_duration_time = [int(x) for x in collapse_duration_time]
            print("Time segment: ", time_list[collapse_duration_time[0]], " ", time_list[collapse_duration_time[1]],
                  " | ", collapse_duration_time[0], " ", collapse_duration_time[1])
            print("Time duration: ", (time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000, " ms")

            print("--------------------")

            temperature_list_original = temperature_list_original[1:inv_radius_channel, 10:-10]
            if collapse_duration_time[0] == 0 or \
                    collapse_duration_time[1] == len(temperature_list_original) - 10 or \
                    len(temperature_list_original) == 0:
                continue

            print('Plotting results and save as images .PNG')
            self.build_temperature_rmaj_single_plot(
                temperature_list_original, self.window_width_val_dur, range(len(temperature_list_original[0])),
                highlight_r_inv, start_offset, median_filter,
                time_limits=collapse_duration_time, discharge=dis)
            # # # # # time_list[10:len(temperature_list_original[0]) + 10]
            # temperature_list_original = temperature_list_original[0:55]
            # self.build_temperature_rmaj_single_plot(
            #     temperature_list_original, self.window_width_val_dur, range(len(temperature_list_original[0])),
            #     highlight_r_inv, start_offset, median_filter,
            #     time_limits=collapse_duration_time, discharge=dis)
            # # # # # # # # # # # # # # # # # # # # # # # #

            print("--------------------")
            print('\n')
            if close_plots == 1:
                plt.close("all")

        return 1

    def build_plots_to_find_inversion_radius(self, start, end, median_filter_window_size, highlight_r_inv, start_offset, close_plots):
        """ -----------------------------------------
            version: 0.3
            desc: build 3d plots, T(r_maj) series, calculate inversion radius
            :return 1
        ----------------------------------------- """

        print('Start detection: inversion radius')

        for dis in range(start, end):

            print("Discharge: " + str(dis + 1))

            if dis == 31:
                continue

            """ Extract data from MATLAB database """
            print('Load data')
            data = dt.Profiling()
            data.load(discharge=dis, source='real')
            # data_public = dt.Profiling(discharge=55, source='public')
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Remove all channels with R_maj = nan """
            print('Remove channels with R_maj = nan')
            r_maj_list = self.processing.dict_to_list(data.channels_pos)
            temperature_list_original = data.temperature_original

            r_maj_list_buffer = []
            temperature_list_buffer = []
            chan_order_buffer = []
            chan_pos_order_buffer = {}
            for t_list_i, t_list in enumerate(temperature_list_original.items()):

                if r_maj_list[t_list_i] > 0:
                    temperature_list_buffer.append(t_list[1])
                    chan_order_buffer.append(t_list[0])
                    chan_pos_order_buffer[t_list[0]] = r_maj_list[t_list_i]
                    r_maj_list_buffer.append(r_maj_list[t_list_i])

            temperature_list_original = temperature_list_buffer
            # print("--------------------")
            # print(chan_pos_order_buffer)
            # print("--------------------")
            r_maj_list = sorted(r_maj_list_buffer)
            temperature_list_original = {chan_order_buffer[i]: k for i, k in enumerate(temperature_list_original)}
            # # # # # # # # # # # # # # # # # # # # # # # #

            print('Sort channels by their own R_maj value')
            """ Sort R_maj index and time lists by R_maj value """
            r_maj_list_indexes = [channel[0] for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1))]
            r_maj_list = [channel[1] for channel in sorted(chan_pos_order_buffer.items(), key=itemgetter(1))]
            channel_order_list = {index: channel[0] for index, channel in enumerate(sorted(chan_pos_order_buffer.items(), key=itemgetter(1)))}
            time_list = data.time_original
            """ Same vars in a.u. units """
            """ Ordering by R_maj """
            temperature_list_original = data.order_by_r_maj(temperature_list_original, chan_pos_order_buffer)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
            Filtering T(t), i.e., smoothing 
            WARNING: do not smooth due to info losses in 3d 
            """
            print('Smoothing channels along timeline')
            temperature_list_original = self.processing.filter(
                temperature_list_original, self.window_width_val_inv, self.window_func)
            # temperature_list_original = temperature_list_original  # skip filtration
            # # # # # # # # # # # # # # # # # # # # # # # #


            """ Calibrate (Normalization on 1) """
            print('Normalizing channels on 1')
            temperature_list_original = data.normalization(temperature_list_original)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Median filtering IMPORTANT to remove outliers T(r_maj) """
            if median_filter_window_size != 0:
                median_filter = str(median_filter_window_size[0]) + 'x' + str(median_filter_window_size[1])
                temperature_list_original = signal_processor.medfilt2d(temperature_list_original, median_filter_window_size)
            else:
                median_filter = 0

            """ 60 - due to the low accuracy after r_inv, which have influence on r_inv detection """
            print('Inversion radius detection')
            temperature_list_original = temperature_list_original[:60, median_filter_window_size[0]:-median_filter_window_size[0]]
            r_maj_list = r_maj_list[:len(temperature_list_original)]
            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list=temperature_list_original,
                                                               window_width=6, std_low_limit=0.01,
                                                               channel_offset=15)

            print("--------------------")
            if inv_radius_channel == 0:
                inv_radius = {
                    'index': 0,
                    'channel': 0,
                    'value': '{:.4f}'.format(0),
                    'value_neighbors': (0, 0)
                }
                print("Detection FAILED")
            else:
                inv_radius = {
                    'index': inv_radius_channel,
                    'channel': r_maj_list_indexes[inv_radius_channel],
                    'value': '{:.4f}'.format(r_maj_list[inv_radius_channel]),
                    'value_neighbors': (r_maj_list[inv_radius_channel - 1], r_maj_list[inv_radius_channel + 1])
                }

                print("Inversion radius index M1: " + str(inv_radius['index']))
                print("Inversion radius channel M1: " + str(inv_radius['channel']))
                print("Inversion radius value M1: " + str(inv_radius['value']))

            print("--------------------")

            print('Plotting results and save as images .PNG')
            # # # # # # # # # # # # # # # # # # # # # # # #
            # self.build_temperature_rmaj_time_3d_surface(temperature_list_original, r_maj_list,
            #                                             time_list, discharge=dis, r_inv=inv_radius_channel)
            # # # # # # # # # # # # # # # # # # # # # # # #
            self.build_temperature_rmaj_series_plot(temperature_list_original[start_offset:], self.window_width_val_inv,
                                                    r_maj_list[start_offset:], highlight_r_inv, discharge=dis, r_inv=inv_radius,
                                                    method="median" + str(median_filter) +
                                                           "_highlight" + str(highlight_r_inv) + "_M1")
            # # # # # # # # # # # # # # # # # # # # # # # #

            print("--------------------")
            print('\n')
            if close_plots == 1:
                plt.close("all")

        return 1

    @staticmethod
    def build_temperature_rmaj_time_3d_surface(temperature_list, r_maj, time_list, **kwargs):
        """ -----------------------------------------
            version: 0.3
            desc: build plot of top view of temperature distribution
            :param temperature_list: 2d list of num
            :param r_maj: 1d list of num
            :param time_list: 1d list of num
            :return 1
        ----------------------------------------- """
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 7)

        # Make data.
        x = time_list[0:len(temperature_list[0])]
        y = r_maj
        x, y = np.meshgrid(x, y)
        z = np.array(temperature_list)

        # COLORMAP HERE "CMRmap", "inferno", "plasma"
        cs = plt.contourf(x, y, z, 100, corner_mask=True, cmap=cm.CMRmap)
        plt.title('Colormap of normalized ECE signals' +
                  ', crash ' + str(kwargs['discharge'] + 1), fontsize=17)
        plt.xlabel('Time (seconds)', fontsize=17)
        plt.ylabel('R maj (m)', fontsize=17)
        # END

        # Add a color bar which maps values to colors.
        cbs = fig.colorbar(cs)
        cbs.ax.set_ylabel('Temperature (a.u.)', fontsize=17)

        directory = 'results/v05/dis' + str(kwargs['discharge'] + 1) + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory + 'dis' + str(kwargs['discharge'] + 1) +
                    '_colormap' +
                    '_total' + str(len(temperature_list)) +
                    '.png')

        return 1

    @staticmethod
    def build_temperature_rmaj_time_3d_surface_perspective(temperature_list, r_maj, time_list, **kwargs):
        """ -----------------------------------------
            version: 0.2
            desc: build plot of movable 3d view of temperature distribution
            :param temperature_list: 2d list of num
            :param r_maj: 1d list of num
            :param time_list: 1d list of num
            :return 1
        ----------------------------------------- """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.set_size_inches(15, 7)

        # Make data.
        x = time_list[0:len(temperature_list[0])]
        y = r_maj
        x, y = np.meshgrid(x, y)
        z = np.array(temperature_list)

        # COLORMAP HERE
        plt.title('Temperature evolution cross JET tokamak, discharge ' + str(kwargs['discharge']), fontsize=17)
        plt.xlabel('Time')
        plt.ylabel('R_maj')
        # END

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.plasma,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_ylim(2.95, 3.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        directory = 'results/extra_normalization/3d_au_0_80/'

        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        #
        # fig.savefig(
        #     directory + 'tokamat_perspective_dis' + str(kwargs['discharge']) + '.png')

        return 1

    @staticmethod
    def build_temperature_rmaj_single_plot(temperature, window_width, time_list, highlight_r_inv,
                                           start_offset, median_filter, **kwargs):
        """ -----------------------------------------
            version: 0.3
            :param temperature: 2d array of filtered and calibrated temperature (list of nums)
            :param temperature_original: 2d of original ordered temperature (list of nums)
            :param window_name: str value of window of recently wind. filtration
            :param channels_pos: 1d array of channel position values
            :param window_width: int width of window of recently wind. filtration
            :param time_list: 1d array of nums
            :param channel_to_compare: int index for array with channels
            :param channel_order_list: 1d array with ordered channels by own value
            :return 1
        -----------------------------------------
        desc: Build single plot T(t) with fixed r_maj
        CAUTION: inverting plasma radius is near 48/49 channel (in ordered unit list)
        CAUTION: channel_to_check means temperature set, ordered by r_maj
        """

        fig, axes = plt.subplots()
        fig.set_size_inches(15, 7)

        """ Time limits of collapse """
        temperature_list_buffer = []
        for channel in range(0, len(temperature), 5):
            temperature_list_buffer.append(temperature[channel])
            axes.plot(
                time_list[start_offset:-start_offset],
                temperature[channel, start_offset:-start_offset],
                color='b'
            )

        max_temp = max(map(max, temperature_list_buffer))
        min_temp = min(map(min, temperature_list_buffer))

        collapse_duration_txt = 0
        if kwargs['time_limits'] != 0 and kwargs['time_limits'][0] > 0 and kwargs['time_limits'][1] < len(time_list):
            if highlight_r_inv == 1:
                rect = patches.Rectangle((time_list[kwargs['time_limits'][0]], min_temp),
                                         (time_list[kwargs['time_limits'][1]] - time_list[kwargs['time_limits'][0]]),
                                         max_temp - min_temp, linewidth=0, edgecolor='r', facecolor='r', alpha=0.2)

                axes.add_patch(rect)

            collapse_duration_txt = '{:.4f}'.format((time_list[kwargs['time_limits'][1]] -
                                                     time_list[kwargs['time_limits'][0]]) * 1000) if kwargs['time_limits'] != 0 else 0

        axes.set(xlabel='Time (seconds)', ylabel='T (a.u.)',
                 title='Channels series, '
                       + 'wind. width ' + str(window_width) +
                        '\nCollapse duration = ' + str(collapse_duration_txt) + "ms")


        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                     axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(17)

        axes.grid()

        directories = [
            'results/v05/dis' + str(kwargs['discharge'] + 1) + '/',
            'results/v05/T_time/'
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

            fig.savefig(directory + 'dis' + str(kwargs['discharge'] + 1) +
                        '_T_time_series_w' + str(window_width) +
                        '_median' + str(median_filter) + '_highlight' + str(highlight_r_inv) +
                        '.png')

        return 1

    @staticmethod
    def build_temperature_rmaj_series_plot(temperature_list, window_width, r_maj, highlight_r_inv, **kwargs):
        """ -----------------------------------------
            version: 0.3
            :param temperature_list: 2d list of num
            :param window_width: num val from wind filtering
            :param r_maj: 1d list of num
            :return 1
        -----------------------------------------
        desc: Build multiple plots T(r_maj)
        with various fixed time
        """

        # fig, axes = plt.subplots()
        fig = plt.figure()
        axes = fig.add_subplot(111)
        fig.set_size_inches(15, 7)

        # # # # # # # # # # # # # # # # # # # # # # # #
        """ Fix determining minmax """
        plot_limit = len(temperature_list[0])
        temperature_list_buffer = []
        for time, temperature in enumerate(np.transpose(temperature_list)):
            if time in range(0, plot_limit, 100):
                temperature_list_buffer.append(temperature)

        label_limit = len(temperature_list)
        for time, temperature in enumerate(np.transpose(temperature_list)):
            """ 
            Labeling each channel (in ordered range) 
            for T(R_maj) plot on the very beginning instant
            """
            if time == 0:

                """ Create double grid """
                axes.minorticks_on()
                axes.grid(which='minor', alpha=0.2)
                axes.grid(which='major', alpha=0.5)

                """ Create labels for every point on plot """
                # labels = range(label_limit)
                #
                # # order_ticks = []
                # for label, x, y in zip(labels, r_maj, temperature):
                #
                #     pos_offset = (0, 20)
                #     if label in range(0, label_limit):
                #         # order_ticks.append(x)
                #         plt.annotate(
                #             label,
                #             xy=(x, y), xytext=pos_offset,
                #             textcoords='offset points', ha='center', va='center',
                #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5'))
                #
                # # ax2 = axes.twiny()
                # # ax2.set_xlim(axes.get_xlim())
                # # ax2.set_xticks(order_ticks)
                # # ax2.set_xticklabels(order_ticks)

                if highlight_r_inv == 1 and kwargs['r_inv']['index'] != 0:
                    rect = patches.Rectangle((kwargs['r_inv']['value_neighbors'][0], min(map(min, temperature_list_buffer))),
                                             (kwargs['r_inv']['value_neighbors'][1] - kwargs['r_inv']['value_neighbors'][0]),
                                             max(map(max, temperature_list_buffer)) - min(map(min, temperature_list_buffer)),
                                             linewidth=3, edgecolor='r', facecolor='r', alpha=0.2)
                    axes.add_patch(rect)

            """ Plot all temperature sets T(r_maj) """
            if time in range(0, plot_limit, 100):
                axes.plot(
                    r_maj,
                    temperature
                )

        # # # # # # # # # # # # # # # # # # # # # # # #

        # axes.set_xlim(min(r_maj), max(r_maj))
        axes.set(ylabel='T (a.u.)', xlabel='R maj (m)',
                 title='T(R_maj) series '
                       'in various time instants\n'
                       'R_inv: channel = ' + str(kwargs['r_inv']['channel']) +
                       ', value = ' + str(kwargs['r_inv']['value']) + 'm')

        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                     axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(17)

        method = ("_" + str(kwargs['method'])) if 'method' in kwargs else ''

        directory = 'results/v05/dis' + str(kwargs['discharge'] + 1) + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory + 'dis' + str(kwargs['discharge'] + 1) +
                    '_T_Rmaj_series_w' + str(window_width) +
                    str(method) +
                    '.png')

        return 1
