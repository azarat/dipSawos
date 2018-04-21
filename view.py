import controller as dt
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

# IMPORTANT to be declared for 3d plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class ViewData:

    def __init__(self):

        """
        I think here will be the main program
        """

        channel_from = 1
        channel_to = 97
        window_width_val = 101
        public_window_width_val = 10

        """ Extract data from MATLAB database """
        self.processing = dt.PreProfiling()
        data = dt.Profiling(discharge=25, channel=(channel_from, channel_to),
                            source='real')
        data_public = dt.Profiling(discharge=25, channel=(channel_from, channel_to),
                                   source='public')
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Ordering by R_maj """
        temperature_ordered_list = data.order_by_r_maj(
            data.temperature_original,
            channel_from, channel_to)
        public_temperature_ordered_list = data.order_by_r_maj(
            data_public.temperature_original,
            channel_from, channel_to)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ 
        Show difference, in %, between 
        original T(t) and smoothed, i.e., filtered T(t)
        with different window functions
        """
        # self.info_losses_wind_func(temperature_ordered_list, data.winListNames, 47)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Calibrate """
        calibrate_temperature_list = data.calibrate(
            temperature_ordered_list, public_temperature_ordered_list,
            window_width_val, public_window_width_val,
            channel_from, channel_to)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """
        Filtering T(t), i.e., smoothing 
        triang          # minimum info save
        blackmanharris  # middle info save
        flattop         # maximum info save
        'boxcar', 'blackman', 'hamming', 'hann',
        'bartlett', 'parzen', 'bohman', 'nuttall', 'barthann'
        """
        temperature_list = self.processing.filter(
            calibrate_temperature_list, window_width_val, 'triang')
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Filtering T(r_maj) WARNING: lose too many info """
        # temperature_list = self.processing.dict_to_list(temperature_list)
        # temperature_list_transpose = self.processing.list_to_dict(np.transpose(temperature_list))
        # temperature_list = self.processing.filter(temperature_list_transpose, 20, 'triang')
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Singe plot with one of window func """
        # self.build_temperature_rmaj_single_plot(temperature_list[27])
        # # # # # # # # # # # # # # # # # # # # # # # #

        r_maj = [channel[1] for channel in sorted(data.channels_pos.items(), key=itemgetter(1))]
        temperature_list = self.processing.dict_to_list(temperature_list)

        self.build_temperature_rmaj_series_plot(temperature_list, public_temperature_ordered_list, r_maj)
        # self.build_temperature_rmaj_time_3d_surface(temperature_list, r_maj)

    def build_temperature_rmaj_time_3d_surface(self, temperature_list, r_maj):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        # print(len(temperature_list))
        # print(len(r_maj))
        X = np.arange(0, len(temperature_list[1]))
        Y = r_maj[0:len(r_maj) - 1]
        X, Y = np.meshgrid(X, Y)
        Z = np.array(temperature_list)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(0.0, 0.5)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf)

        plt.show()

    def build_temperature_rmaj_single_plot(self, temperature):
        # # # # # # # # # # # # # # # # # # # # # # # #
        # LOGIC TEST. Build single plot T(t) with fixed r_maj
        # !!! CAUTION: inverting plasma radius is near 48 channel
        # CAUTION: channel_to_check means temperature set, ordered by r_maj
        fig, axes = plt.subplots()
        fig.set_size_inches(15, 8)

        axes.plot(
            range(0, len(temperature)),
            temperature
        )

        axes.set(xlabel='R maj (num order only)', ylabel='T (eV)',
                 title='JET tokamat temperature evolution')
        axes.grid()

        plt.show()

    def build_temperature_rmaj_series_plot(self, temperature_list, public_temperature_ordered_list, r_maj):

        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(15, 8)

        # # # # # # # # # # # # # # # # # # # # # # # #
        # WORKING AREA. Build multiple plots T(r_maj)
        # with various fixed time
        for time, temperature in enumerate(np.transpose(temperature_list)):
            if time in range(0, 4000, 100):
                axes[0].plot(
                    r_maj[0:80],
                    # range(0, len(temperature[0:80])),
                    temperature[0:80]
                )

        for time, temperature in enumerate(np.transpose(public_temperature_ordered_list)):
            if time in range(0, 100, 2):
                axes[1].plot(
                    r_maj[0:80],
                    # range(0, len(temperature[0:80])),
                    temperature[0:80]
                )
        # # # # # # # # # # # # # # # # # # # # # # # #

        axes[0].set(ylabel='T (eV)',
               title='JET tokamat temperature evolution')
        axes[1].set(xlabel='R maj (num order only)', ylabel='T (eV)')
        axes[0].grid()
        axes[1].grid()

        plt.show()
        # fig.savefig('results/filters/d25_c55_w81.png')

    def info_losses_wind_func(self, temperature_original, analyze_window, channel_to_compare):

        """
        Compare all types of window function

        WARNING: Data manipulations below should be moved to controller
        """
        sum_deviation = []

        for index, name in enumerate(analyze_window):

            filtered_temperature = self.processing.list_to_dict(temperature_original)
            filtered_temperature = self.processing.filter(filtered_temperature, 81, name)
            filtered_temperature = self.processing.dict_to_list(filtered_temperature)[channel_to_compare]

            filtered_temperature_size = len(filtered_temperature)
            cut_temperature_original = temperature_original[channel_to_compare][0:filtered_temperature_size]

            sum_deviation_array = np.fabs((cut_temperature_original / filtered_temperature) - 1)
            sum_deviation.append(
                (np.sum(sum_deviation_array) / len(sum_deviation_array)) * 100,
            )

        """ Plot compare deviation """
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)

        y_pos = np.arange(len(analyze_window))
        performance = sum_deviation

        ax.barh(y_pos, performance)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(analyze_window)

        ax.set(xlabel='Losses in %', ylabel='Window type',
               title='Loss of info, 55 channel, 25 discharge, wind. width 81')
        ax.grid()
        ax.set_xlim(left=2.2)

        plt.show()
        # fig.savefig('results/filters/compare_wind_funcs_d25_c55_w81.png')
