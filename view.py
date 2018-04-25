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

        """ Sort R_maj and time lists by R_maj value """
        r_maj = [channel[1] for channel in sorted(data.channels_pos.items(), key=itemgetter(1))]
        channel_order_list = {index: channel[0] for index, channel in enumerate(sorted(data.channels_pos.items(), key=itemgetter(1)))}
        time_list = [data.time_original[channel[1]] for channel in channel_order_list.items()]

        """ Ordering by R_maj """
        temperature_ordered_list = data.order_by_r_maj(
            data.temperature_original,
            channel_from, channel_to)
        public_temperature_ordered_list = data.order_by_r_maj(
            data_public.temperature_original,
            channel_from, channel_to)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Calibrate """
        calibrate_temperature_list = data.calibrate(
            temperature_ordered_list, public_temperature_ordered_list,
            window_width_val, public_window_width_val,
            channel_from, channel_to)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ 
        CAUTION: for different channels losses could be VERy different
        so it would be preferable to check losses for each specific case

        Show difference, in %, between 
        original T(t) and smoothed, i.e., filtered T(t)
        with different window functions
        """
        self.info_losses_wind_func(
            channels_pos=data.channels_pos,
            window_width=81,
            temperature_original=calibrate_temperature_list,
            analyze_window=data.winListNames,
            channel_to_compare=22,
            time_list=time_list)
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
        # self.build_temperature_rmaj_single_plot(temperature_list[49])
        # # # # # # # # # # # # # # # # # # # # # # # #

        temperature_list = self.processing.dict_to_list(temperature_list)

        # self.build_temperature_rmaj_series_plot(temperature_list, public_temperature_ordered_list, r_maj)
        # self.build_temperature_rmaj_time_3d_surface(temperature_list, r_maj)

        plt.show()

    @staticmethod
    def build_temperature_rmaj_time_3d_surface(temperature_list, r_maj):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        x = np.arange(0, len(temperature_list[1]))
        y = r_maj[0:len(r_maj) - 1]
        x, y = np.meshgrid(x, y)
        z = np.array(temperature_list)

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(0.0, 0.5)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf)

        return 1

    @staticmethod
    def build_temperature_rmaj_single_plot(temperature):

        """
        Build single plot T(t) with fixed r_maj
        CAUTION: inverting plasma radius is near 48/49 channel (in ordered unit list)
        CAUTION: channel_to_check means temperature set, ordered by r_maj
        """

        fig, axes = plt.subplots()
        fig.set_size_inches(15, 8)

        axes.plot(
            range(0, len(temperature)),
            temperature
        )

        axes.set(xlabel='R maj (num order only)', ylabel='T (eV)',
                 title='JET tokamat temperature evolution')
        axes.grid()

        return 1

    @staticmethod
    def build_temperature_rmaj_series_plot(temperature_list, public_temperature_ordered_list, r_maj):

        """
        Build multiple plots T(r_maj)
        with various fixed time
        """

        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(15, 8)

        # # # # # # # # # # # # # # # # # # # # # # # #
        for time, temperature in enumerate(np.transpose(temperature_list)):
            if time in range(0, 4000, 100):
                axes[0].plot(
                    # r_maj[0:80],
                    range(0, len(temperature[0:80])),
                    temperature[0:80]
                )

        for time, temperature in enumerate(np.transpose(public_temperature_ordered_list)):
            if time in range(0, 100, 2):
                axes[1].plot(
                    # r_maj[0:80],
                    range(0, len(temperature[0:80])),
                    temperature[0:80]
                )
        # # # # # # # # # # # # # # # # # # # # # # # #

        axes[0].set(ylabel='T (eV)',
               title='JET tokamat temperature evolution')
        axes[1].set(xlabel='R maj (num order only)', ylabel='T (eV)')
        axes[0].grid()
        axes[1].grid()

        # fig.savefig('results/filters/d25_c55_w81.png')

        return 1

    def info_losses_wind_func(self, **kwargs):

        """
        Compare all types of window function
        """

        temperature_original = kwargs['temperature_original']

        if temperature_original and type(temperature_original) is dict:
            temperature_original = self.processing.dict_to_list(temperature_original)

        sum_deviation = self.processing.calculate_deviation(
            kwargs['window_width'],
            temperature_original,
            kwargs['analyze_window'],
            kwargs['channel_to_compare'])

        """ Plot compare deviation """
        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(10, 8)

        y_pos = np.arange(len(kwargs['analyze_window']))

        axes[0].barh(y_pos, sum_deviation)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(kwargs['analyze_window'])

        axes[0].set(xlabel='RMSD (eV)', ylabel='Window type',
                    title='Losses of info, ' + str(kwargs['channel_to_compare'])
                          + ' channel, 25 discharge, '
                            'wind. width ' + str(kwargs['window_width']) +
                          ', R_maj = ' + str(kwargs['channels_pos'][kwargs['channel_to_compare']]))
        axes[0].grid()
        axes[0].set_xlim(
            left=(min(sum_deviation) - min(sum_deviation) / 100),
            right=(max(sum_deviation) + min(sum_deviation) / 100))

        """ Plot compared T(t) """
        axes[1].plot(
            # range(0, len(temperature_original[kwargs['channel_to_compare'] - 1])),
            kwargs['time_list'][kwargs['channel_to_compare']],
            temperature_original[kwargs['channel_to_compare']]
        )
        axes[1].set(xlabel='Time (seconds)', ylabel='Temperature (eV)',
                    title='')

        axes[1].grid()

        """ Align label position """
        for ax in axes:
            label = ax.xaxis.get_label()
            x_lab_pos, y_lab_pos = label.get_position()
            label.set_position([1.0, y_lab_pos])
            label.set_horizontalalignment('right')

        # fig.savefig('results/filters/compare_wind_funcs_d25_c55_w81.png')

        return 1
