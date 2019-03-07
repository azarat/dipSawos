import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as signal_processor


class QProfileModule:
    center = 3.02  # SHOULD be calculated

    internal_channel_index = 0
    internal_channel_position = 0

    def get_x_mix(self, data, data_public, inv_radius, collapse_duration_time):
        """ -----------------------------------------
            version: 0.9
            desc: 
            :return
        ----------------------------------------- """

        channel_index = np.asarray([channel[0] for channel in data])
        channel_position = np.asarray([channel[1] for channel in data])
        temperature_matrix = np.asarray([channel[2] for channel in data])
        temperature_matrix_public = data_public

        self.channel_index = channel_index
        self.channel_position = channel_position

        # channel_position_o = channel_position
        channel_position = [v-self.center for v in channel_position]
        channel_position = [((np.power(v, 2)) * (v / np.abs(v))) for v in channel_position]

        xc_index_order = self.nearest_channel_index(0, channel_position, channel_index)

        """ !OUTDATED! """
        xc_index = xc_index_order['index']
        # xs = self.find_xs(temperature_matrix_public, temperature_matrix, channel_position, channel_index, xc_index, inv_radius, collapse_duration_time[1])
        # xs = np.power((np.sqrt(xs) + self.center), 2)
        # xs_index_order = self.nearest_channel_index(np.sqrt(xs), channel_position, channel_index)

        """ Normalization to PPF """
        temperature_matrix = self.normalization_ppf(temperature_matrix, temperature_matrix_public, channel_index)

        temperature_pre_post = {
            'pre': np.transpose(temperature_matrix)[collapse_duration_time[0]],
            'post': np.transpose(temperature_matrix)[collapse_duration_time[1]]
        }

        """ Median det. killer (median filtration) """
        temperature_pre_post['pre'] = signal_processor.medfilt(temperature_pre_post['pre'], 31)
        temperature_pre_post['post'] = signal_processor.medfilt(temperature_pre_post['post'], 31)

        """ Kill detalization (window filtration) """
        # window = signal_processor.get_window('triang', 10)
        # temperature_pre_post['pre'] = signal_processor.convolve(temperature_pre_post['pre'], window, mode='valid')
        # temperature_pre_post['post'] = signal_processor.convolve(temperature_pre_post['post'], window, mode='valid')

        """ DEBUG """
        # plt.close('all')
        # fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(15, 7)
        # ax.plot(channel_position, temperature_pre_post['pre'], label="pre")
        # ax.plot(channel_position, temperature_pre_post['post'], label="post")
        # plt.axvline(x=float(np.power(float(inv_radius['position']) - self.center, 2)), label="Inv rad", color="black")
        # plt.legend()
        #
        # fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(15, 7)
        # ax.plot(channel_position_o, temperature_pre_post['pre'], label="pre")
        # ax.plot(channel_position_o, temperature_pre_post['post'], label="post")
        # plt.axvline(x=float(inv_radius['position']), label="Inv rad", color="black")
        # plt.legend()
        #
        # plt.show()
        # exit()

        xs = self.find_xs_ppf(temperature_pre_post, channel_position)
        xs_index_order = self.nearest_channel_index(xs, channel_position, channel_index)

        x_2 = self.integrate_eiler(xs, xs_index_order, channel_position, temperature_pre_post, inv_radius)

        x_mix = x_2[-1]
        r_mix = np.sqrt(x_mix) + self.center

        return r_mix

    def find_xs_ppf(self, temperature_pre_post, channel_position):

        temperature_xc = self.temperature_interpolation(0, channel_position, temperature_pre_post['post'])

        diff = [temperature_xc - v for v in reversed(temperature_pre_post['pre'])]

        channel_before = 0
        channel_after = 0
        for i, v in enumerate(diff):
            if v <= 0:
                channel_after = len(diff) - i
                channel_before = len(diff) - i - 1
                break

        position_gap = np.linspace(channel_position[channel_before], channel_position[channel_after], 100)
        temperature_gap = np.linspace(temperature_pre_post['pre'][channel_before], temperature_pre_post['pre'][channel_after], 100)

        diff_gap = [temperature_xc-v for v in temperature_gap]

        point_cs = 0

        for i, v in enumerate(diff_gap):
            if v >= 0:
                point_cs = i
                break

        position_cs = position_gap[point_cs]

        return position_cs

    @staticmethod
    def normalization_ppf(temperature_matrix, temperature_matrix_public, channel_index):
        """ -----------------------------------------
            version: 0.9
            desc:
            :return
        ----------------------------------------- """

        temperature_matrix_norm = []
        for order, index in enumerate(channel_index):
            temperature_matrix_norm.append(temperature_matrix[order] * temperature_matrix_public[index][1])

        return temperature_matrix_norm

    @staticmethod
    def temperature_interpolation(inter_position, channel_position, temperature):

        # channel_position = [np.power(v, 2) for v in channel_position]
        diff = [v-inter_position for v in channel_position]

        channel_after = 0
        channel_before = 0
        for i, v in enumerate(diff):
            if v >= 0:
                channel_before = i-1
                channel_after = i
                break

        """ Difference between siblings temperature"""
        diff_temperature = temperature[channel_after] - temperature[channel_before]
        a = np.abs(diff_temperature)

        """ Difference between siblings position (distance) """
        b = channel_position[channel_after] - channel_position[channel_before]
        c = np.sqrt( np.power(a, 2) + np.power(b, 2) )

        """ Difference between interpoint and channel after (distance) """
        b2 = channel_position[channel_after] - inter_position
        """ Difference between interpoint and channel before (distance) """
        b1 = b - b2

        """ Hypotenuse between channel before and interpoint """
        c1 = b1 * c / b

        """ Difference between interpoint temperature and one of siblings temperature """
        h1 = np.sqrt( np.power(c1, 2) - np.power(b1, 2) )

        intertemperature = 0
        if diff_temperature <= 0:
            intertemperature = temperature[channel_before] - h1
        elif diff_temperature > 0:
            intertemperature = temperature[channel_before] + h1

        return intertemperature

    def find_xs(self, temperature_matrix_public, temperature_matrix, channel_position, channel_index, xc_index, inv_radius, crash_end):
        """ -----------------------------------------
            !OUDATED!
            version: 0.9
            desc: t => temperature
                  x = r^2
            :param temperature_matrix_public:
            :param temperature_matrix:
            :param channel_position:
            :param channel_index:
            :return: position of xs [float]
        ----------------------------------------- """

        """ convert ndarray to list """
        channel_index = [v for v in channel_index]

        t_center_precrash_norm = 1
        """ we should calculate xs from the magnet center (not from the center of torus) """
        x_inv = pow(float(inv_radius['position'])-self.center, 2)

        t_center_precrash_public = temperature_matrix_public[xc_index][3]
        xc_order = channel_index.index(xc_index)
        t_center_postcrash_norm = temperature_matrix[xc_order][crash_end]

        t_inv_public = temperature_matrix_public[inv_radius['index']][3]

        t_inv_norm = t_inv_public / t_center_precrash_public
        h = 1 - (t_inv_norm / t_center_postcrash_norm)
        g = h * (x_inv / t_center_precrash_norm)

        xs = x_inv - g

        return xs

    @staticmethod
    def nearest_channel_index(position, channel_position, channel_index):
        """ -----------------------------------------
            version: 0.9
            desc: find the index of the channel which is nearest to the position
            :param position:
            :param channel_position:
            :param channel_index:
            :return: index of nearest channel to the input position value [int]
        ----------------------------------------- """

        channel_position_diff = [abs(x - position) for x in channel_position]
        diff_min = min(channel_position_diff)
        position_channel_order = channel_position_diff.index(diff_min)

        channel_index_and_order = {
            'index': channel_index[position_channel_order],
            'order': position_channel_order
        }

        return channel_index_and_order

    @staticmethod
    def integration_function(temperature_plus, temperature_1, temperature_2):
        return (temperature_1 - temperature_plus) / (temperature_2 - temperature_plus)

    def integrate_eiler(self, xs, xs_index_order, channel_position, temperature_pre_post, inv_radius):

        n_steps = 100
        integration_boundaries = np.linspace(xs, 0, n_steps)

        y = 0
        h = 0
        temperature_1 = 0
        temperature_2 = 0
        temperature_plus = 0
        function_value = 0
        x_1 = []
        f = []
        T_1 = []
        T_2 = []
        T_plus = []
        x_plus = []
        h_array = []
        x_2 = []
        for i, inter_position in enumerate(integration_boundaries):

            if i == 0:
                y = xs
                function_value = -1
            else:

                h = (inter_position - integration_boundaries[i-1])
                y = y + h * function_value

                temperature_1 = self.temperature_interpolation(inter_position, channel_position, temperature_pre_post['pre'])
                temperature_2 = self.temperature_interpolation(y, channel_position, temperature_pre_post['pre'])
                temperature_plus = self.temperature_interpolation((y - inter_position), channel_position, temperature_pre_post['post'])
                function_value = self.integration_function(temperature_plus, temperature_1, temperature_2)

            if np.isnan(inter_position) or np.isnan(y - inter_position) \
                    or np.isnan(y) or np.isnan(temperature_1) or np.isnan(temperature_2) or np.isnan(temperature_plus):
                break

            h_array.append(h)
            x_2.append(y)
            x_plus.append(y - inter_position)
            x_1.append(inter_position)
            f.append(function_value)
            T_1.append(temperature_1)
            T_2.append(temperature_2)
            T_plus.append(temperature_plus)

            """ DEBUG """
            print("-------------------------STEP ", i)
            print("x1 = ", inter_position, " --- xs = ", xs, " --- y = ", y, " --- x+ = ", (y - inter_position))
            print("-------------------------")
            print("T1 = ", temperature_1, " --- T2 = ", temperature_2, " --- T+ = ", temperature_plus)
            print("-------------------------")
            print("f = ", function_value)
            print("-------------------------")

        return x_2

    def get_psi_rmix(self, psi_r, r_mix):
        """ -----------------------------------------
             version: 0.10
             desc: find value psi_r in a point r_mix
                   psi_r ~ r^2
         ----------------------------------------- """

        diff_r = self.channel_position - r_mix

        left_channel_order = 0
        right_channel_order = 0
        for i, d in enumerate(diff_r):
            if d > 0:
                left_channel_order = i-1
                right_channel_order = i
                break

        """ Find interposition via triangular equations """
        """ Katet """
        b = self.channel_position[right_channel_order] - self.channel_position[left_channel_order]
        """ Part of the same katet """
        b1 = r_mix - self.channel_position[left_channel_order]
        """ Another katet """

        """ DEBUG """
        # plt.close('all')
        # fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(15, 7)
        # ax.plot(psi_r)
        #
        # plt.show()
        # exit()


        a = psi_r[right_channel_order] - psi_r[left_channel_order]
        """ Hypotenuza """
        c = np.sqrt(np.power(a, 2) + np.power(b, 2))
        """ Triangular part of psi we looking for """
        a1 = b1 * b / c

        psi_rmix = 0
        if psi_r[left_channel_order] < psi_r[right_channel_order]:
            psi_rmix = psi_r[left_channel_order] + a1
        else:
            psi_rmix = psi_r[left_channel_order] - a1


        return psi_rmix

    @property
    def channel_index(self):
        return self.internal_channel_index

    @channel_index.setter
    def channel_index(self, value):
        self.internal_channel_index = value

    @property
    def channel_position(self):
        return self.internal_channel_position

    @channel_position.setter
    def channel_position(self, value):
        self.internal_channel_position = value
