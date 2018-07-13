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
    window_width_val = 100
    # window_width_val = 500  # time smoothing
    window_width_rad_val = 10  # radius smoothing
    window_func = 'triang'
    boundary = (0.75, 1.5)
    # boundary = (-100, 100)

    def __init__(self):
        print("VERSION: 0.3")
        # self.build_plots_to_find_inversion_radius()
        self.build_plots_to_find_collapse_time_duration()
        # self.extra_normalization()
        # self.export_inversion_radius()
        # self.ml_find_inv_radius()
        # self.ml_find_collapse_duration()
        # self.export_collapse_duration()

    @staticmethod
    def ml_find_collapse_duration():
        """ -----------------------------------------
            version: 0.3
            desc: export collapse_duration to train neural network
                  save to mat files
                  every channel have sum of time points
            :return 1
        ----------------------------------------- """

        """ Extract data from MATLAB database """
        controller = dt.MachineLearning()
        data = controller.ml_load('machine_learning/train/train_ECE_data_Rinv.mat')

        dis_count = len(data['ece_data'][0, :])  # 31
        list_dis_all = data['discharge'][0, :]
        # # # # # # # # # # # # # # # # # # # # # # # #

        list_dis = []
        export_data_dis = []
        for dis in range(dis_count):
            print('Discharge: ', dis)

            temperature_list_all = data['ece_data'][0, :][dis]
            temperature_list = np.transpose(temperature_list_all)[:60]

            """ Find collapse duration """
            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list, 5, 0.01)

            if inv_radius_channel == 0:
                continue

            # collapse_duration_time = dt.FindCollapseDuration().collapse_duration(temperature_list, 0.2, inv_radius_channel, 0.01)
            # print(collapse_duration_time)
            # plt.plot(temperature_list[15])
            # plt.show()
            # exit()

            list_dis.append(list_dis_all[dis])
            export_data = []
            for i, ch in enumerate(temperature_list_all):
                t_list_export = []
                for t in ch:
                    t_list_export.append(t)
                t_list_export.append(inv_radius_channel)
                export_data.append(t_list_export)

            export_data_dis.append(export_data)
            print("Inversion radius M1: " + str(inv_radius_channel))

        data_to_save = {
            'ece_data': {
                'discharge': list_dis,
                'signal': np.array(export_data_dis)
            },
            'description': 'ECE_data with R_inv indicator at each channel 0/1 based on ECE_data_tS500_n1_mF5_uDR.mat'}
        io.savemat('machine_learning/train/train_ECE_data_Rinv.mat', data_to_save)

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

    @staticmethod
    def export_inversion_radius():
        """ -----------------------------------------
            version: 0.3
            desc: export inv radius to train neural network
                  save to mat files
            :return 1

            7dis without r_inv
        ----------------------------------------- """

        """ Extract data from MATLAB database """
        controller = dt.MachineLearning()
        data = controller.ml_load('machine_learning/ECE_data_tS500_n1_mF5_uDR.mat')

        dis_count = len(data['ece_data'][0, :])  # 32
        list_dis_all = data['discharge'][0, :]
        # # # # # # # # # # # # # # # # # # # # # # # #

        list_dis = []
        export_data_dis = []
        for dis in range(dis_count):
            print('Discharge: ', dis)

            temperature_list_all = data['ece_data'][0, :][dis]
            temperature_list = np.transpose(temperature_list_all)[:60]

            """ Find Inversion Radius """
            """ Method 1 - identify closest channel """
            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list, 5, 0.01)

            if inv_radius_channel == 0:
                continue

            list_dis.append(list_dis_all[dis])
            export_data = []
            for i, ch in enumerate(temperature_list_all):
                t_list_export = []
                for t in ch:
                    t_list_export.append(t)
                t_list_export.append(inv_radius_channel)
                export_data.append(t_list_export)

            export_data_dis.append(export_data)
            print("Inversion radius M1: " + str(inv_radius_channel))

        data_to_save = {
            'ece_data': {
                'discharge': list_dis,
                'signal': np.array(export_data_dis)
            },
            'description': 'ECE_data with R_inv indicator at each channel 0/1 based on ECE_data_tS500_n1_mF5_uDR.mat'}
        io.savemat('machine_learning/train/train_ECE_data_Rinv.mat', data_to_save)

        return 1

    def extra_normalization(self):
        """ -----------------------------------------
            version: 0.3
            desc:
            :return 1

            status: IN DEV
            3 4 12 8 18 21 22
        ----------------------------------------- """

        temperature_list_dis = []
        list_dis = []
        start_time = time.time()

        for dis in range(3, 5):
            start_time_internal = time.time()

            print("Discharge: " + str(dis))

            if dis == 31 or dis in range(11, 24) or dis in range(47, 64):
                continue

            """ Extract data from MATLAB database """
            data = dt.Profiling()
            data.load(discharge=dis, source='real')
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Sort R_maj index and time lists by R_maj value """
            r_maj = [channel[1] for channel in sorted(data.channels_pos.items(), key=itemgetter(1))][self.channel_from:self.channel_to]
            channel_order_list = {index: channel[0] for index, channel in enumerate(sorted(data.channels_pos.items(), key=itemgetter(1)))}
            time_list = data.time_original
            """ Same vars in a.u. units """

            """ Ordering by R_maj """
            temperature_list_original = data.order_by_r_maj(data.temperature_original)[self.channel_from:self.channel_to]
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Filtering T(t), i.e., smoothing """
            temperature_list = self.processing.filter(
                temperature_list_original, self.window_width_val, self.window_func)
            # temperature_list = temperature_list_original  # skip filtration

            """ Calibrate (Normalization on 1) """
            temperature_list = data.normalization(temperature_list)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ IMPORTANT to remove sawtooth behavior T(r_maj) """
            # temperature_list = self.processing.filter(
            #     np.transpose(temperature_list), self.window_width_rad_val, self.window_func)
            temperature_list = np.transpose(temperature_list)  # skip filtration
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Median filtering IMPORTANT to remove outliers T(r_maj) """
            window_size = 5
            outliers_list = np.array(temperature_list).tolist()
            temperature_list_filtered = []

            for t_list in outliers_list:
                temperature_filtered = []
                for ii in range(0, len(t_list), window_size):
                    temperature_filtered += self.processing.median_filtered(np.asanyarray(t_list[ii: ii + window_size]), 3).tolist()
                temperature_list_filtered.append(temperature_filtered)
            # temperature_list_filtered = temperature_list  # skip filtration
            # # # # # # # # # # # # # # # # # # # # # # # #

            # plt.plot(temperature_list_filtered[500], label="500")
            # plt.plot(temperature_list_filtered[1000], label="1000")
            # plt.plot(temperature_list_filtered[1500], label="1500")
            # plt.plot(temperature_list_filtered[2000], label="2000")
            # plt.legend()
            # plt.show()
            temperature_list_dis.append(temperature_list_filtered)
            list_dis.append(dis)

            print("--- %s seconds ---" % (time.time() - start_time_internal))

        data_to_save = {
            'ece_data': {
                'discharge': list_dis,
                'signal': temperature_list_dis
            },
            'description': 'ECE_data_timeSmoothed500_normalized1_medianFiltered5_unsortedDisRemoved'}
        print(len(temperature_list_dis))
        # io.savemat('machine_learning/ECE_data_tS500_n1_mF5_uDR.mat', data_to_save)
        print("------ %s seconds ------" % (time.time() - start_time))

        return 1

    def build_plots_to_find_collapse_time_duration(self):
        """ -----------------------------------------
            version: 0.2
            desc: build plots with collapse duration time-points
            :return 1
        ----------------------------------------- """

        for dis in range(1, 64):

            print("Discharge: " + str(dis))

            if dis == 31 or dis == 26 or dis in range(11, 24) or dis in range(47, 64):
                continue

            """ Extract data from MATLAB database """
            data = dt.Profiling()
            data.load(discharge=dis, source='real')
            # data_public = dt.Profiling(discharge=55, source='public')
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Sort R_maj index and time lists by R_maj value """
            r_maj = [channel[1] for channel in sorted(data.channels_pos.items(), key=itemgetter(1))][self.channel_from:self.channel_to]
            channel_order_list = {index: channel[0] for index, channel in enumerate(sorted(data.channels_pos.items(), key=itemgetter(1)))}
            time_list = data.time_original
            """ Same vars in a.u. units """

            """ Ordering by R_maj """
            temperature_list_original = data.order_by_r_maj(data.temperature_original)[self.channel_from:self.channel_to]
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Calibrate (Normalization on 1) """
            temperature_list_clear = data.normalization(temperature_list_original)
            temperature_list = temperature_list_clear
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Median filtering IMPORTANT to remove outliers T(r_maj) """
            window_size = 5
            outliers_list = np.array(temperature_list).tolist()
            temperature_list_filtered = []

            for t_list in np.transpose(outliers_list):
                temperature_filtered = []
                for ii in range(0, len(t_list), window_size):
                    temperature_filtered += self.processing.median_filtered(np.asanyarray(t_list[ii: ii + window_size]), 3).tolist()
                temperature_list_filtered.append(temperature_filtered)
            temperature_list = np.transpose(temperature_list_filtered)
            temperature_list_filtered = temperature_list[:60]
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
             Filtering T(t), i.e., smoothing 
             WARNING: do not smooth due to info losses in 3d 
             """
            temperature_list = self.processing.filter(
                temperature_list, self.window_width_val, self.window_func)
            # temperature_list = temperature_list_original  # skip filtration

            """ IMPORTANT to remove sawtooth behavior T(r_maj) """
            temperature_list = self.processing.filter(
                np.transpose(temperature_list), self.window_width_rad_val, self.window_func)
            temperature_list = np.transpose(temperature_list)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
            Cut outliers
            WARNING: Have influence on inv. rad. detection (more cut => less val of outlier need)
            """
            # temperature_list = data.outlier_filter_std_deviation(temperature_list, 2, 1)
            # temperature_list = data.outlier_filter(temperature_list, self.boundary)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Find Inversion Radius """
            """ Method 1 - identify closest channel """
            # # # # # # # # # # # # # # # # # # # # # # # #
            inv_radius_channel_index = dt.FindInvRadius().inv_radius(temperature_list, 5, 0.01)
            inv_radius_val = r_maj[inv_radius_channel_index] if inv_radius_channel_index > 0 else 0
            inv_radius_channel = ('{:.4f}'.format(inv_radius_val), inv_radius_channel_index)
            print("Inversion radius M1: " + str(inv_radius_channel))
            # # # # # # # # # # # # # # # # # # # # # # # #

            # inv_radius_channel_index = 0
            # inv_radius_channel = (0, 0)

            # self.build_temperature_rmaj_series_plot(temperature_list, self.window_width_val, r_maj[0:len(temperature_list)], discharge=dis, inv_rad_c=inv_radius_channel, method="M1")
            # self.build_temperature_rmaj_series_plot(temperature_list, self.window_width_val, r_maj[0:len(temperature_list)], discharge=dis, inv_rad_c=inv_radius_channel)

            # plt.plot(temperature_list[15])
            # plt.show()
            # print(len(temperature_list))
            # exit()

            # # # # # # # # # # # # # # # # # # # # # # # #
            collapse_duration_time = dt.FindCollapseDuration().collapse_duration(temperature_list, 6, inv_radius_channel_index, 1.03)
            collapse_duration_time = [int(x+(self.window_width_val * 0.7)) for x in collapse_duration_time]
            print("Time segment: ", time_list[collapse_duration_time[0]], " ", time_list[collapse_duration_time[1]])
            print("Time duration: ", (time_list[collapse_duration_time[1]] - time_list[collapse_duration_time[0]]) * 1000, " ms")
            # # # # # # # # # # # # # # # # # # # # # # # #

            temperature_list = temperature_list[:60]
            self.build_temperature_rmaj_time_3d_surface(temperature_list_filtered, range(len(temperature_list_filtered)), range(len(temperature_list_filtered[0])), window_width=self.window_width_val, discharge=dis, time_limits=collapse_duration_time)
            # self.build_temperature_rmaj_time_3d_surface_perspective(temperature_list, range(len(temperature_list)), range(len(temperature_list[0])), discharge=dis)

            # for i in range(0, len(np.transpose(temperature_list)), 100):
            #     plt.plot(np.transpose(temperature_list)[i])
            # plt.show()
            # exit()

            # # # # # # # # # # # # # # # # # # # # # # # #
            # self.build_temperature_rmaj_single_plot(
            #     temperature_list[:70], self.window_func, self.window_width_val, range(len(temperature_list[0])),
            #     time_limits=collapse_duration_time, discharge=dis)
            # # # # # # # # # # # # # # # # # # # # # # # #

            # plt.close("all")
            # plt.show()
            # exit()

        plt.show()

        return 1

    def build_plots_to_find_inversion_radius(self):
        """ -----------------------------------------
            version: 0.2
            desc: build 3d plots, T(r_maj) series and single plots
            :return 1
        ----------------------------------------- """

        for dis in range(1, 64):

            print("Discharge: " + str(dis))

            if dis == 31 or dis in range(11, 24) or dis in range(47, 64):
                continue

            """ Extract data from MATLAB database """
            data = dt.Profiling()
            data.load(discharge=dis, source='real')
            # data_public = dt.Profiling(discharge=55, source='public')
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Sort R_maj index and time lists by R_maj value """
            r_maj = [channel[1] for channel in sorted(data.channels_pos.items(), key=itemgetter(1))][self.channel_from:self.channel_to]
            channel_order_list = {index: channel[0] for index, channel in enumerate(sorted(data.channels_pos.items(), key=itemgetter(1)))}
            time_list = data.time_original
            """ Same vars in a.u. units """

            """ Ordering by R_maj """
            temperature_list_original = data.order_by_r_maj(data.temperature_original)[self.channel_from:self.channel_to]
            # public_temperature_ordered_list = data.order_by_r_maj(data_public.temperature_original)[channel_from:channel_to]
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
            Filtering T(t), i.e., smoothing 
            WARNING: do not smooth due to info losses in 3d 
            """
            temperature_list = self.processing.filter(
                temperature_list_original, self.window_width_val, self.window_func)
            # temperature_list = temperature_list_original  # skip filtration

            """ Calibrate (Normalization on 1) """
            # temperature_list_original = data.normalization(temperature_list_original)
            temperature_list = data.normalization(temperature_list)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ IMPORTANT to remove sawtooth behavior T(r_maj) """
            # temperature_list = self.processing.filter(
            #     np.transpose(temperature_list), self.window_width_rad_val, self.window_func)
            # temperature_list = np.transpose(temperature_list)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """
            Cut outliers
            WARNING: Have influence on inv. rad. detection (more cut => less val of outlier need)
            """
            # temperature_list_original = data.outlier_filter(temperature_list_original, boundary)
            temperature_list = data.outlier_filter_std_deviation(temperature_list, 2, 1)
            temperature_list = data.outlier_filter(temperature_list, self.boundary)
            # # # # # # # # # # # # # # # # # # # # # # # #

            """ Find Inversion Radius """
            """ Method 1 - identify closest channel """
            inv_radius_channel = dt.FindInvRadius().inv_radius(temperature_list, 5, 0.01)
            inv_radius_val = r_maj[inv_radius_channel] if inv_radius_channel > 0 else 0
            inv_radius_channel = ('{:.4f}'.format(inv_radius_val), inv_radius_channel)
            print("Inversion radius M1: " + str(inv_radius_channel))
            self.build_temperature_rmaj_series_plot(temperature_list, self.window_width_val, r_maj[0:len(temperature_list)], discharge=dis, inv_rad_c=inv_radius_channel, method="M1")
            """ Method 2 - identify intersection position """
            inv_radius_channel = dt.FindInvRadius().inv_radius_intersection(temperature_list, 5, 0.00001, r_maj)
            print("Inversion radius M2: " + str(inv_radius_channel))
            self.build_temperature_rmaj_series_plot(temperature_list, self.window_width_val, r_maj[0:len(temperature_list)], discharge=dis, inv_rad_c=inv_radius_channel, method="M2")

            # self.build_temperature_rmaj_time_3d_surface(temperature_list, r_maj[0:len(temperature_list)], time_list[0:len(temperature_list[0])], window_width=self.window_width_val, discharge=dis, inv_rad_c=inv_radius_channel)
            # self.build_temperature_rmaj_time_3d_surface_perspective(temperature_list, r_maj, time_list, discharge=dis)
            # self.build_temperature_rmaj_single_plot(
            #     temperature=temperature_list,
            #     temperature_original=temperature_list_original,
            #     channels_pos=r_maj,
            #     window_width=window_width_val,
            #     channel_to_compare=3,
            #     time_list=time_list,
            #     channel_order_list=channel_order_list,
            #     window_name=window_func)

            plt.close("all")

        # plt.show()

        return 1

    @staticmethod
    def build_temperature_rmaj_time_3d_surface(temperature_list, r_maj, time_list, **kwargs):
        """ -----------------------------------------
            version: 0.2
            desc: build plot of top view of temperature distribution
            :param temperature_list: 2d list of num
            :param r_maj: 1d list of num
            :param time_list: 1d list of num
            :return 1
        ----------------------------------------- """
        fig, ax = plt.subplots(1, 1)
        # fig = plt.figure()
        fig.set_size_inches(15, 7)

        # Make data.
        x = time_list[0:len(temperature_list[0])]
        # x = range(0,len(temperature_list[0]))
        y = r_maj
        x, y = np.meshgrid(x, y)
        z = np.array(temperature_list)

        # COLORMAP HERE "CMRmap", "inferno", "plasma"
        cs = plt.contourf(x, y, z, 50, corner_mask=True, cmap=cm.CMRmap)
        plt.title('Temperature evolution' +
                  ', discharge ' + str(kwargs['discharge']), fontsize=17)
        plt.xlabel('Time', fontsize=17)
        plt.ylabel('R_maj', fontsize=17)
        # END

        max_temp = max(r_maj)
        min_temp = 0

        rect = patches.Rectangle((time_list[kwargs['time_limits'][0]], min_temp),
                                 (time_list[kwargs['time_limits'][1]] - time_list[kwargs['time_limits'][0]]),
                                 max_temp - min_temp, linewidth=0, edgecolor='r', facecolor='g', alpha=0.3)

        ax.add_patch(rect)


        # Add a color bar which maps values to colors.
        cbs = fig.colorbar(cs)
        cbs.ax.set_ylabel('Temperature (eV)', fontsize=17)

        directory = 'results/extra_normalization/3d_au_0_80/'

        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        #
        # fig.savefig(directory + 'tokamat_colormap_dis' + str(kwargs['discharge']) + '_w' + str(kwargs['window_width']) + '.png')

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
    def build_temperature_rmaj_single_plot( temperature, window_name, window_width, time_list, **kwargs):
        """ -----------------------------------------
            version: 0.2
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

        # axes.plot(
        #     # range(0, len(temperature_original[channel_to_compare])),
        #     time_list[0:len(temperature_original[channel_to_compare])],
        #     temperature_original[channel_to_compare],
        #     alpha=0.5,
        #     color='c'
        # )

        """ Time limits of collapse """
        mix_temp_list = []
        for t_list in temperature:
            for t in t_list:
                mix_temp_list.append(t)

        # axes.plot(
        #     [time_list[kwargs['time_limits'][0]], time_list[kwargs['time_limits'][0]]],
        #     [min_temp, max_temp],
        #     color='r'
        # )
        #
        # axes.plot(
        #     [time_list[kwargs['time_limits'][1]], time_list[kwargs['time_limits'][1]]],
        #     [min_temp, max_temp],
        #     color='r'
        # )

        max_temp = max(mix_temp_list)
        min_temp = min(mix_temp_list)

        if kwargs['time_limits'] != 0:
            rect = patches.Rectangle((time_list[kwargs['time_limits'][0]], min_temp),
                                     (time_list[kwargs['time_limits'][1]] - time_list[kwargs['time_limits'][0]]),
                                     max_temp - min_temp, linewidth=0, edgecolor='r', facecolor='r', alpha=0.2)

            axes.add_patch(rect)
        # # # # # # # # # # # # # # # # # # # # # #

        for channel in range(0, 65, 5):
            axes.plot(
                # range(0, len(temperature[channel_to_compare])),
                time_list[0:len(temperature[channel])],
                temperature[channel],
                color='b'
            )

        collapse_duration_txt = '{:.4f}'.format((time_list[kwargs['time_limits'][1]] -
                                                 time_list[kwargs['time_limits'][0]]) * 1000) if kwargs['time_limits'] != 0 else 0

        axes.set(xlabel='Time (seconds)', ylabel='T (eV)',
                 title='Signal, "'
                       + window_name + '" wind. func., '
                       + ' ' + str(kwargs['discharge']) + ' discharge, ' +
                       'wind. width ' + str(window_width) +
                        ', Collapse duration = ' + str(collapse_duration_txt) + "ms")


        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                     axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(17)

        axes.grid()

        directory = 'results/extra_normalization/T_time_series/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory + 'dis' + str(kwargs['discharge']) +
                    '_T_time_series_w' + str(window_width) + '.png')

        return 1

    @staticmethod
    def build_temperature_rmaj_series_plot(temperature_list, window_width, r_maj, **kwargs):
        """ -----------------------------------------
            version: 0.2
            :param temperature_list: 2d list of num
            :param window_width: num val from wind filtering
            :param r_maj: 1d list of num
            :return 1
        -----------------------------------------
        desc: Build multiple plots T(r_maj)
        with various fixed time
        """

        fig, axes = plt.subplots()
        fig.set_size_inches(15, 7)

        # # # # # # # # # # # # # # # # # # # # # # # #
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
                labels = range(label_limit)

                for label, x, y in zip(labels, r_maj, temperature):
                    if label < 50:
                        pos_offset = (0, 20)
                    else:
                        pos_offset = (0, -20)

                    if label in range(0, label_limit, 2):
                        plt.annotate(
                            label,
                            xy=(x, y), xytext=pos_offset,
                            textcoords='offset points', ha='center', va='center',
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5'))

            """ Plot all temperature sets T(r_maj) """
            plot_limit = len(temperature_list[0])
            if time in range(0, plot_limit, 100):
                axes.plot(
                    r_maj,
                    temperature
                )

        # # # # # # # # # # # # # # # # # # # # # # # #

        axes.set(ylabel='T (eV)', xlabel='R maj (m)',
                 title='T(r_maj) series '
                       'in various time instants, '
                       'win. width ' + str(window_width) +
                       ', inv. rad. ' + str(kwargs['inv_rad_c']))

        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
                     axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(17)

        method = ("_" + str(kwargs['method'])) if 'method' in kwargs else ''

        directory = 'results/ml/T_Rmaj_series' + str(method) + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory + 'dis' + str(kwargs['discharge']) +
                    '_T_Rmaj_series_w' + str(window_width) + '.png')

        return 1
