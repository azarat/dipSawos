import controller as dt
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class ViewData:

    def show_filtered_plots(self):

        """
        Just show list of plots with window filtering
        I like triang, blackmanharris, flattop windows
        """

        data = dt.DataPreparing()
        data.get_data(discharge=25, channel=55, source='real', filterType='multiple', window_width=81, window_name='')

        print(data.winListNames)

        fig, ax = plt.subplots()
        for index, name in enumerate(data.winListNames):
            if name in ['hamming', 'barthann', 'bartlett',
                        'hann', 'nuttall', 'parzen',
                        'boxcar', 'bohman', 'blackman']:
                continue
            ax.plot(data.time, data.temperature[name] + (index * 200), label=name)

        ax.plot(data.time_original, data.temperature_original - 1000, label='original')

        ax.set(xlabel='time (s)', ylabel='T (eV with shifts)',
               title='JET tokamat temperature evolution, 55 channel, 25 discharge, wind. width 81')
        ax.grid()

        plt.legend()
        plt.show()
        # fig.savefig('results/filters/d25_c55_w81.png')

    def info_losses_wind_func(self):

        """
        Compare all types of window function
        """

        data = dt.DataPreparing()
        data.get_data(discharge=25, channel=55, source='real', filterType='multiple', window_width=81, window_name='')

        """
        WARNING: Data manipulations below should be moved to controller
        """
        analyze_window = data.winListNames
        filtered_temp_size, cut_time, cut_temp_original = ({} for i in range(3))
        sum_deviation = []

        for index, name in enumerate(analyze_window):

            filtered_temp_size.update({
                name: len(data.temperature[name])
            })
            cut_time.update({
                name: data.time[0:filtered_temp_size[name]]
            })
            cut_temp_original.update({
                name: data.temperature_original[0:filtered_temp_size[name]]
            })
            sum_deviation_array = np.fabs((cut_temp_original[name] / data.temperature[name]) - 1)
            sum_deviation.append(
                (np.sum(sum_deviation_array) / len(sum_deviation_array)) * 100,
            )
            # print(str(name) + ': ' + str(sum_deviation[name]))

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

    def get_single_filtered_plot(self):

        """
        Singe plot with one of window func

        triang          # minimum info save
        blackmanharris  # middle info save
        flattop         # maximum info save
        'boxcar', 'blackman', 'hamming', 'hann',
        'bartlett', 'parzen', 'bohman', 'nuttall', 'barthann'
        """

        data = dt.DataPreparing()
        data.get_data(discharge=25, channel=55, source='real', filterType='single', window_width=81, window_name='triang')

        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)

        ax.plot(data.time, data.temperature)

        ax.set(xlabel='time (s)', ylabel='T (eV with shifts)',
               title='JET tokamat temperature evolution, 55 channel, 25 discharge, wind. width 81')
        ax.grid()

        plt.show()
        # fig.savefig('results/filters/d25_c55_w81.png')

    def get_3d_plot(self):

        """
        3D plot ...
        """

        channel_from = 1
        channel_to = 97

        data = dt.Profiling(discharge=25, channel=(channel_from, channel_to),
                            source='public')

        temperature_ordered = []

        # print(data.temperature[1])
        for channel in sorted(data.channels_pos.items(), key=itemgetter(1)):
            if channel[0] in range(channel_from, channel_to):
                temperature_ordered.append(
                    data.temperature_original[channel[0]]
                )

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(0, len(temperature_ordered[1]))
        Y = np.arange(0, len(temperature_ordered))
        X, Y = np.meshgrid(X, Y)
        Z = np.array(temperature_ordered)

        # print(len(temperature_ordered[1]))
        # print(len(temperature_ordered))

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(0.0, 0.5)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def get_2d_plot_ordered_by_rmaj(self):

        """
        get_2d_plot_ordered_by_rmaj ...
        I think here will be the main program
        """

        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(15, 8)

        channel_from = 1
        channel_to = 97
        window_width_val = 101
        public_window_width_val = 10

        # # # # # # # # # # # # # # # # # # # # # # # #
        # Extract data from MATLAB database
        data = dt.Profiling(discharge=25, channel=(channel_from, channel_to),
                            source='real')
        data_public = dt.Profiling(discharge=25, channel=(channel_from, channel_to),
                                   source='public')

        # # # # # # # # # # # # # # # # # # # # # # # #
        # Ordering by R_maj
        temperature_ordered_list = data.order_by_r_maj(
            data.temperature_original,
            channel_from, channel_to)
        public_temperature_ordered_list = data.order_by_r_maj(
            data_public.temperature_original,
            channel_from, channel_to)

        # # # # # # # # # # # # # # # # # # # # # # # #
        # Calibrate
        calibrate_temperature_list = data.calibrate(
            temperature_ordered_list, public_temperature_ordered_list,
            window_width_val, public_window_width_val,
            channel_from, channel_to)

        # # # # # # # # # # # # # # # # # # # # # # # #
        # Filtering T(t), i.e., smoothing
        temperature_list = data.filter(
            calibrate_temperature_list, window_width_val, 'triang')

        # # # # # # # # # # # # # # # # # # # # # # # #
        # Filtering T(r_maj) WARNING: lose too many info
        # temperature_list = data.dict_to_list(temperature_list)
        # temperature_list_transpose = data.list_to_dict(np.transpose(temperature_list))
        # temperature_list = data.filter(temperature_list_transpose, 20, 'triang')

        # # # # # # # # # # # # # # # # # # # # # # # #
        # LOGIC TEST. Build single plot T(t) with fixed r_maj
        # !!! CAUTION: inverting plasma radius is near 48 channel
        # channel_to_check = 27
        # axes[0].plot(
        #     range(0, len(temperature_list[channel_to_check])),
        #     temperature_list[channel_to_check]
        # )

        # # # # # # # # # # # # # # # # # # # # # # # #
        # WORKING AREA. Build multiple plots T(r_maj)
        # with various fixed time
        r_maj = [channel[1] for channel in sorted(data.channels_pos.items(), key=itemgetter(1))]
        temperature_list = data.dict_to_list(temperature_list)

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
