import controller as dt
import matplotlib.pyplot as plt
import numpy as np


class ViewData:

    def show_filtered_plots(self):

        """
        Just show list of plots with window filtering
        I like triang, blackmanharris, flattop windows
        """

        data = dt.DataPreparing()
        data.get_data(discharge=25, channel=55, source='real', filterType='multiple', windowW=81)

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
        data.get_data(discharge=25, channel=55, source='real', filterType='multiple', windowW=81)

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

