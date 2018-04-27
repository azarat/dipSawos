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
        window_width_val = 500
        public_window_width_val = 10
        window_func = 'triang'

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
        # self.info_losses_wind_func(
        #     channels_pos=r_maj,
        #     window_width=window_width_val,
        #     temperature_original=calibrate_temperature_list,
        #     analyze_window=data.winListNames,
        #     channel_to_compare=60,
        #     time_list=time_list,
        #     channel_order_list=channel_order_list)
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
            calibrate_temperature_list, window_width_val, window_func)
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Filtering T(r_maj) WARNING: lose too many info """
        # temperature_list = self.processing.dict_to_list(temperature_list)
        # temperature_list_transpose = self.processing.list_to_dict(np.transpose(temperature_list))
        # temperature_list = self.processing.filter(temperature_list_transpose, 20, 'triang')
        # # # # # # # # # # # # # # # # # # # # # # # #

        """ Singe plot with one of window func """
        # self.build_temperature_rmaj_single_plot(
        #     temperature=temperature_list,
        #     temperature_original=calibrate_temperature_list,
        #     channels_pos=r_maj,
        #     window_width=window_width_val,
        #     channel_to_compare=50,
        #     time_list=time_list,
        #     channel_order_list=channel_order_list,
        #     window_name=window_func)
        # # # # # # # # # # # # # # # # # # # # # # # #

        temperature_list = self.processing.dict_to_list(temperature_list)

        # self.build_temperature_rmaj_series_plot(temperature_list, public_temperature_ordered_list,
        #                                         window_width_val, r_maj)
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

    def build_temperature_rmaj_single_plot(self,
                                           temperature, temperature_original, window_name,
                                           channels_pos, window_width, time_list,
                                           channel_to_compare, channel_order_list):

        """
        Build single plot T(t) with fixed r_maj
        CAUTION: inverting plasma radius is near 48/49 channel (in ordered unit list)
        CAUTION: channel_to_check means temperature set, ordered by r_maj
        """

        fig, axes = plt.subplots()
        fig.set_size_inches(15, 8)

        axes.plot(
            # range(0, len(temperature_original[channel_to_compare])),
            time_list[channel_to_compare][0:len(temperature_original[channel_to_compare])],
            temperature_original[channel_to_compare],
            alpha=0.5,
            color='c'
        )

        axes.plot(
            # range(0, len(temperature[channel_to_compare])),
            time_list[channel_to_compare][0:len(temperature[channel_to_compare])],
            temperature[channel_to_compare],
            color='b'
        )

        axes.set(xlabel='Time (seconds)', ylabel='T (eV)',
                 title='Original signal vs filtered, "'
                       + window_name + '" wind. func., '
                       + str(channel_order_list[channel_to_compare]) + ' channel, 25 discharge, '
                       'wind. width ' + str(window_width) +
                       ', R_maj = ' + str(channels_pos[channel_to_compare]))
        axes.grid()

        self.align_axes(axes)

        fig.savefig('results/filters/single_plots/single_d25_c' +
                    str(channel_order_list[channel_to_compare]) +
                    '_w' + str(window_width) +
                    '_WF' + window_name +
                    '.png')

        return 1

    def build_temperature_rmaj_series_plot(self, temperature_list, public_temperature_ordered_list, window_width, r_maj):

        """
        Build multiple plots T(r_maj)
        with various fixed time
        """

        fig, axes = plt.subplots()
        fig.set_size_inches(15, 8)

        # # # # # # # # # # # # # # # # # # # # # # # #
        for time, temperature in enumerate(np.transpose(temperature_list)):
            """ 
            Labeling each channel (in ordered range) 
            for T(R_maj) plot on the very beginning instant
            """
            if time == 0:
                labels = range(80)

                for label, x, y in zip(labels, r_maj[0:80], temperature[0:80]):
                    if label <= 23:
                        pos_offset = (-10, 10)
                    elif label in range(23, 50):
                        pos_offset = (10, 10)
                    else:
                        pos_offset = (-10, -10)

                    if label in range(0, 80, 2):
                        plt.annotate(
                            label,
                            xy=(x, y), xytext=pos_offset,
                            textcoords='offset points', ha='center', va='center',
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            if time in range(0, 4000, 100):
                axes.plot(
                    r_maj[0:80],
                    # range(0, len(temperature[0:80])),
                    temperature[0:80]
                )
        # # # # # # # # # # # # # # # # # # # # # # # #

        axes.set(ylabel='T (eV)', xlabel='R maj (m)',
                 title='Temperature changes inside toroid '
                       'in various time instants '
                       'with time smoothing, win. width ' + str(window_width))
        axes.grid()

        # self.align_axes(axes)

        fig.savefig('results/filters/d25_T_Rmaj_with_labeled_points_c0080_w' + str(window_width) + '.png')

        return 1

    def info_losses_wind_func(self, **kwargs):

        """
        Compare all types of window function
        """

        temperature_original = kwargs['temperature_original']
        time = kwargs['time_list'][kwargs['channel_to_compare']]

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
                    title='Losses of info, ' + str(kwargs['channel_order_list'][kwargs['channel_to_compare']])
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

        custom_grid_time = [time[i] for i in range(len(time)) if (i % kwargs['window_width'] == 0)]

        axes[1].set_xticks(custom_grid_time, minor=True)
        axes[1].xaxis.grid(True, which='minor')

        axes[1].set(xlabel='Time (seconds)', ylabel='Temperature (eV)',
                    title='')

        # axes[1].grid()

        self.align_axes(axes)

        fig.savefig('results/filters/RMSD/RMSD_d25_c' +
                    str(kwargs['channel_order_list'][kwargs['channel_to_compare']]) +
                    '_w' + str(kwargs['window_width']) +
                    '.png')

        return 1

    @staticmethod
    def align_axes(axes):
        if axes is not list or tuple:
            axes = [axes]

        for ax in axes:
            label = ax.xaxis.get_label()
            x_lab_pos, y_lab_pos = label.get_position()
            label.set_position([1.0, y_lab_pos])
            label.set_horizontalalignment('right')

            label = ax.yaxis.get_label()
            x_lab_pos, y_lab_pos = label.get_position()
            label.set_position([x_lab_pos, .95])
            label.set_verticalalignment('top')

        return 1

