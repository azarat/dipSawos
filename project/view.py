import project.controllers.q_profile as q_profile
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

import csv
import json
# from pprint import pprint

import scipy.io as io
import time


class ViewData:
    internal_shots = 0
    internal_psi_r = 0
    internal_q_database = 0
    internal_input_parameters = 0

    q_profile = q_profile.QProfile()
    processing = dt.PreProfiling()
    channel_from = 0
    channel_to = 92
    window_width_val_dur = 3  # time smoothing
    window_width_val_inv = 500  # time smoothing to find inv radius
    window_width_rad_val = 5  # radius smoothing
    window_func = 'triang'
    boundary = (0.75, 1.5)

    close_plots = 1
    discharge = 3

    def __init__(self):
        print("--------------------------------------VERSION: 0.10")

        self.input_parameters = self.read_input_parameters()
        median_filter_window_size = (self.input_parameters['median_filter_window_size'],
                                     self.input_parameters['median_filter_window_size'])

        if self.close_plots == 0:
            """ Single dis. Offset for DB numbering """
            dis_end = self.input_parameters['discharges']['start']
            dis_start = dis_end - 1
        else:
            """ Range of dis """
            dis_start, dis_end = 0, 10

        """ Outside loop """
        results = []
        db_controller = dt.Controller()
        db = db_controller.load()

        if type(self.shots) is int:
            self.shots = db_controller.shots

        for dis in range(dis_start, dis_end):

            print("------Load public data------DISCHARGE: ", self.shots[dis_start])
            data_public = self.prepare_data(database=db, source="public", dis=dis_start,
                                            median_filter_window_size=median_filter_window_size,
                                            window_filter_width_size=0, window_function=self.window_func)

            print("------Load real data------")
            data = self.prepare_data(database=db, source="real", dis=dis_start,
                                     median_filter_window_size=median_filter_window_size,
                                     window_filter_width_size=0, window_function=self.window_func)

            """ 60 - due to the low accuracy after r_inv, which have influence on r_inv detection """
            print("------Inversion radius detection------")
            temperature_matrix = np.asarray([channel[2] for channel in data])[:80, 9: -9]

            print('Smoothing channels along timeline')
            temperature_matrix_smooth = self.processing.filter(
                temperature_matrix, self.window_width_val_inv, self.window_func)

            print('R_inv detecting')
            r_maj_list = np.asarray([channel[1] for channel in data])[:len(temperature_matrix_smooth)]
            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list=temperature_matrix_smooth,
                                                               window_width=6, std_low_limit=0.01,
                                                               channel_offset=15)

            inv_radius = 0
            if inv_radius_channel > 0:
                inv_radius = {
                    'index': np.asarray([channel[0] for channel in data])[inv_radius_channel],
                    'sorted_order': inv_radius_channel,
                    'position': '{:.4f}'.format(r_maj_list[inv_radius_channel]),
                    'position_neighbors': (r_maj_list[inv_radius_channel - 1], r_maj_list[inv_radius_channel + 1])
                }

                print(' ')
                print("Inversion radius index: " + str(inv_radius['index']))
                print("Inversion radius order number: " + str(inv_radius['sorted_order']))
                print("Inversion radius position: " + str(inv_radius['position']))

            else:
                print("FAILED")

            """ Identifying collapse duration """
            print("------Identifying collapse duration------")
            collapse_duration_time = dt.FindCollapseDuration().collapse_duration([],
                                                                                 temperature_matrix, 6,
                                                                                 inv_radius_channel, 1.03,
                                                                                 median_filter_window_size)

            time_list = data[0][3]

            collapse_duration_time = [int(x) for x in collapse_duration_time]
            print("Time segment: ", time_list[collapse_duration_time[0]], " ", time_list[collapse_duration_time[1]],
                  "ms | ", collapse_duration_time[0], " ", collapse_duration_time[1], " point numbers")
            print("Time duration: ", (time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000,
                  " ms")

            print("--------------------")

            if collapse_duration_time[0] == 0 or \
                    collapse_duration_time[1] == len(temperature_matrix) or \
                    len(temperature_matrix) == 0:
                print("FAILED")

            """ !IN DEV! """
            """ Q profile """
            # r_mix = self.q_profile.get_x_mix(data, data_public, inv_radius, collapse_duration_time)
            # psi_rmix = self.q_profile.get_psi_rmix(self.psi_r, r_mix)
            #
            # """ DEBUG """
            # plt.close('all')
            # fig, ax = plt.subplots(1, 1)
            # fig.set_size_inches(15, 7)
            # for q in self.q_database:
            #     ax.plot(q)
            #
            # plt.show()
            # exit()

            # # # # # # # # # # # # # # # # # # # # # # # #
            """ !OLD VER! """
            """ Inversion radius """
            # self.build_plots_to_find_inversion_radius(start=dis_start, end=dis_end, median_filter_window_size=(3, 3),
            #                                           highlight_r_inv=1, start_offset=1, close_plots=self.close_plots)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ !OLD VER! """
            """ Collapse duration """
            # results = False
            # results = self.build_plots_to_find_collapse_time_duration(start=dis_start, end=dis_end,
            #                                                           median_filter_window_size=(9, 9),
            #                                                           highlight_r_inv=1, close_plots=self.close_plots)
            # # # # # # # # # # # # # # # # # # # # # # # #
            """ !OLD VER! """
            """ Colormap overview """
            # self.build_plots_colormap(start=dis_start, end=dis_end, median_filter_window_size=(3, 3),
            #                           start_offset=4, end_offset=-30, close_plots=self.close_plots)
            # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # #

            result = [self.shots[dis_start],
                      time_list[collapse_duration_time[0]],
                      time_list[collapse_duration_time[1]],
                      (time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000,
                      inv_radius['index'],
                      inv_radius['position']]

            results.append(result)

            print("--------------------------------------COMPLETE")
            print("")

        if self.close_plots == 0:
            plt.show()
        else:
            self.write_into_file(results)
        # # # # # # # # # # # # # # # # # # # # # # # #

    @staticmethod
    def read_input_parameters():
        """ -----------------------------------------
            version: 0.10
            desc: read input parameters from JSON file
            :return [list]
        ----------------------------------------- """

        with open('input.json') as f:
            input_parameters = json.load(f)

        return input_parameters


    def prepare_data(self, database, source, dis, median_filter_window_size, window_filter_width_size, window_function):
        """ -----------------------------------------
            version: 0.9
            desc: extract data from MatLab database, clear from channels with incomplete info,
                  sort channels, normalized on 1, smooth with different methods and combine into one object
            :return if source real: array with full info about each channel [int, float, array of float]
                    if source public: 2D array of floats
        ----------------------------------------- """

        """ Extract data from MATLAB database """
        print('Load data')
        data = dt.Profiling()
        data.assign(database=database, discharge=dis, source=source)

        if type(self.psi_r) is int:
            self.psi_r = data.psi_r

        if source == 'public':

            temperature_matrix = data.temperature_original
            return temperature_matrix

        else:

            """ Remove all channels with R_maj = nan """
            print('Remove channels with R_maj = nan')
            r_maj_list = self.processing.dict_to_list(data.channels_pos)
            temperature_matrix = data.temperature_original
            time_list = data.time

            temperature_matrix_buffer = []
            chan_order_buffer = []
            channels = {}
            for t_list_i, t_list in enumerate(temperature_matrix.items()):

                if r_maj_list[t_list_i] > 0:
                    """ temperature list of channel without r_maj = nan """
                    temperature_matrix_buffer.append(t_list[1])

                    """ channel number without r_maj = nan """
                    chan_order_buffer.append(t_list[0])

                    """ channel position (r_maj) without r_maj = nan (assigned to channel number) """
                    channels[t_list[0]] = r_maj_list[t_list_i]

            """ matrix without r_maj = nan """
            temperature_matrix = temperature_matrix_buffer
            """ transform back from list to dict """
            temperature_matrix = {chan_order_buffer[i]: k for i, k in enumerate(temperature_matrix)}
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Sort channel number, channel position, channel temperature by R_maj value """
            print('Sort channels by their own R_maj value')
            channels_sorted = sorted(channels.items(), key=itemgetter(1))

            channel_number = [channel[0] for channel in channels_sorted]
            channel_position = [channel[1] for channel in channels_sorted]
            temperature_matrix = [temperature_matrix[channel[0]] for channel in channels_sorted]

            # print(channel_number[60])
            # exit()

            """
            Filtering T(t), i.e., smoothing 
            WARNING: much info losses 
            """
            # if window_filter_width_size > 0:
            #     print('Smoothing channels along timeline')
            #     temperature_matrix = self.processing.filter(
            #         temperature_matrix, window_filter_width_size, window_function)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Calibrate (Normalization on 1) """
            print('Normalizing channels on 1')
            temperature_matrix = data.normalization(temperature_matrix)
            # # # # # # # # # # # # # # # # # # # # # # # #

            # fig, ax = plt.subplots(1, 1)
            # fig.set_size_inches(15, 7)
            #
            # plt.title('Original data of ECE KK3 JPF, discharge 86459, channel 36', fontsize=17)
            # plt.xlabel('Time (seconds)', fontsize=17)
            # plt.ylabel('T', fontsize=17)
            #
            # ax.plot(time_list, temperature_matrix[60])
            # # ax.plot(np.transpose(temperature_matrix)[500])
            # # ax.plot(np.transpose(temperature_matrix)[-500])
            #
            # plt.show()
            # exit()

            """ Median filtering. IMPORTANT to remove outliers """
            if median_filter_window_size != 0:
                temperature_matrix = signal_processor.medfilt2d(temperature_matrix, median_filter_window_size)

            """ Combine all data to one object """
            data = [(channel_number[i], channel_position[i], temperature_matrix[i], time_list)
                    for i, c in enumerate(temperature_matrix)]

            return data

    @staticmethod
    def write_into_file(results):
        """ -----------------------------------------
            version: 0.10
            desc: write results in file
            :return 1
        ----------------------------------------- """
        directory = 'results/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + 'output.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Discharge order', 'Discharge JET number', 'Start, ms', 'End, ms', 'Duration, ms',
                                'Inv. radius channel', 'Inv. radius, m'])
            for row_i, row in enumerate(results):
                row = [row_i + 1] + row
                csvwriter.writerow(row)

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
            data.assign(discharge=dis, source='real')
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
            time_list = data.time
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
            ch_start_offset = 0
            ch_end_offset = 55
            self.build_temperature_rmaj_time_3d_surface(temperature_list_original[ch_start_offset:ch_end_offset, start_offset:-start_offset],
                                                        r_maj_list[ch_start_offset:ch_end_offset],
                                                        time_list[start_offset:-start_offset],
                                                        discharge=dis, median_filter=median_filter)

            print("--------------------")
            print('\n')
            if close_plots == 1:
                plt.close("all")

        return 1

    def build_plots_to_find_collapse_time_duration(self, start, end, median_filter_window_size, highlight_r_inv, close_plots):
        """ -----------------------------------------
            version: 0.3
            desc: build plots with collapse duration time-points
            :return 1
        ----------------------------------------- """

        print('Start detection: collapse duration')
        results = []
        for dis in range(start, end):

            print("Discharge: " + str(dis + 1))

            if dis == 31:
                continue

            """ Extract data from MATLAB database """
            print('Load data')
            data = dt.Profiling()
            data.assign(discharge=dis, source='real')
            # data_public = dt.Profiling(discharge=55, source='public')
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Remove all channels with R_maj = nan """
            print('Remove channels with R_maj = nan')
            r_maj_list = self.processing.dict_to_list(data.channels_pos)
            temperature_list_original = data.temperature_original
            shots = data.shots

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
            time_list = data.time
            """ Same vars in a.u. units """
            """ Ordering by R_maj """
            temperature_list_original = data.order_by_r_maj(temperature_list_original, chan_pos_order_buffer)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
            Filtering T(t), i.e., smoothing 
            WARNING: do not smooth due to info losses in 3d 
            """
            print('Smoothing channels along timeline SKIP')
            # temperature_list_original = self.processing.filter(
            #     temperature_list_original, self.window_width_val_dur, self.window_func)
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
            # r_maj_list = r_maj_list[:len(temperature_list_rad)]
            r_maj_list = [x * x for x in r_maj_list]

            # fig, ax = plt.subplots(1, 1)
            # fig.set_size_inches(15, 7)
            # ax.plot(r_maj_list)
            # plt.show()
            # exit()

            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list=temperature_list_rad,
                                                               window_width=6, std_low_limit=0.01,
                                                               channel_offset=15)

            """ 
            Smooth T(r_maj)
            IMPORTANT to remove sawtooth behavior T(r_maj)
            """
            print('Smoothing channels along radius SKIP')
            # temperature_list_original = self.processing.filter(
            #     np.transpose(temperature_list_original), self.window_width_rad_val, self.window_func)
            # temperature_list_original = np.transpose(temperature_list_original)
            # # # # # # # # # # # # # # # # # # # # # # #

            print("--------------------")

            """ Identifying collapse duration """
            print('Determination of collapse duration')
            collapse_duration_time = dt.FindCollapseDuration().collapse_duration(temperature_list_reverse, temperature_list_original, 6, inv_radius_channel, 1.03, median_filter_window_size)
            collapse_duration_time = [int(x) for x in collapse_duration_time]
            print("Time segment: ", time_list[collapse_duration_time[0]], " ", time_list[collapse_duration_time[1]],
                  " | ", collapse_duration_time[0], " ", collapse_duration_time[1])
            print("Time duration: ", (time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000, " ms")

            print("--------------------")

            if collapse_duration_time[0] == 0 or \
                    collapse_duration_time[1] == len(temperature_list_original) or \
                    len(temperature_list_original) == 0:
                continue

            print('Plotting results and save as images .PNG')
            self.build_temperature_rmaj_single_plot(
                temperature_list_original, self.window_width_val_dur,
                time_list,
                # range(len(temperature_list_original[0])),
                highlight_r_inv, median_filter_window_size[0], median_filter,
                time_limits=collapse_duration_time, discharge=dis, inv_radius_channel=inv_radius_channel)

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

            result = []
            result.append(shots[dis])
            result.append(time_list[collapse_duration_time[0]])
            result.append(time_list[collapse_duration_time[1]])
            result.append((time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000)
            result.append(r_maj_list_indexes[inv_radius_channel])
            result.append(r_maj_list[inv_radius_channel])
            results.append(result)
        # return 1
        return results

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
            data.assign(discharge=dis, source='real')
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
            time_list = data.time
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
        cs = plt.contourf(x, y, z, 50, alpha=1, corner_mask=True, cmap=cm.CMRmap)
        plt.title('Colormap of normalized ECE signals' +
                  ', crash ' + str(kwargs['discharge'] + 1), fontsize=17)
        plt.xlabel('Time (seconds)', fontsize=17)
        plt.ylabel('R maj (m)', fontsize=17)
        # END

        # Add a color bar which maps values to colors.
        cbs = fig.colorbar(cs)
        cbs.ax.set_ylabel('Temperature (a.u.)', fontsize=17)

        directory = 'results/v06/dis' + str(kwargs['discharge'] + 1) + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory + 'dis' + str(kwargs['discharge'] + 1) +
                    '_colormap' +
                    '_total' + str(len(temperature_list)) +
                    '.png')

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

        # if kwargs['inv_radius_channel'] > 1:
        #     temperature = temperature[:kwargs['inv_radius_channel']]
        # else:
        #     temperature = temperature[:45]
        temperature = temperature[:kwargs['inv_radius_channel']+10]

        for t_list_index, t_list in enumerate(temperature):
            if t_list_index % 4 == 0:
                axes.plot(
                    time_list[start_offset:-start_offset],
                    t_list[start_offset:-start_offset],
                    color="b"
                )

        """ Time limits of collapse """
        max_temp = np.amax(temperature)
        min_temp = np.amin(temperature)

        collapse_duration_txt = 0
        if kwargs['time_limits'] != 0 and kwargs['time_limits'][0] > 0 and kwargs['time_limits'][1] < len(time_list):
            if highlight_r_inv == 1:
                rect = patches.Rectangle((time_list[kwargs['time_limits'][0]], min_temp),
                                         (time_list[kwargs['time_limits'][1]] - time_list[kwargs['time_limits'][0]]),
                                         max_temp - min_temp, linewidth=0, edgecolor='r', facecolor='r', alpha=0.3)

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
            'results/v06/dis' + str(kwargs['discharge'] + 1) + '/',
            'results/v06/T_time/'
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
        axes.set(ylabel='T (a.u.)', xlabel='R (m)',
                 title='R_inv: channel = ' + str(kwargs['r_inv']['channel']) +
                       ', value = ' + str(kwargs['r_inv']['value']) + 'm')

        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                     axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(17)

        method = ("_" + str(kwargs['method'])) if 'method' in kwargs else ''

        directory = 'results/v06/dis' + str(kwargs['discharge'] + 1) + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory + 'dis' + str(kwargs['discharge'] + 1) +
                    '_T_Rmaj_series_w' + str(window_width) +
                    str(method) +
                    '.png')

        return 1

    @property
    def psi_r(self):
        return self.internal_psi_r

    @psi_r.setter
    def psi_r(self, value):
        self.internal_psi_r = value

    @property
    def q_database(self):
        return self.internal_q_database

    @q_database.setter
    def q_database(self, value):
        self.internal_q_database = value

    @property
    def shots(self):
        return self.internal_shots

    @shots.setter
    def shots(self, value):
        self.internal_shots = value

    @property
    def input_parameters(self):
        return self.internal_input_parameters

    @input_parameters.setter
    def input_parameters(self, value):
        self.internal_input_parameters = value

