import matplotlib.pyplot as plt
import numpy as np


class QProfile:
    center = 2.92  # SHOULD be calculated

    def calculate_q_profile(self, data, data_public, inv_radius, collapse_duration_time):
        """ -----------------------------------------
            version: 0.9
            desc: 
            :return
        ----------------------------------------- """

        channel_index = np.asarray([channel[0] for channel in data])
        channel_position = np.asarray([channel[1] for channel in data])
        temperature_matrix = np.asarray([channel[2] for channel in data])
        temperature_matrix_public = data_public

        xc_index = self.find_center(temperature_matrix, channel_position, channel_index)

        xs = self.find_xs(temperature_matrix_public, temperature_matrix, channel_position, channel_index, xc_index, inv_radius, collapse_duration_time[1])

        self.integrate_eiler(xs, xs)

        return 1

    @staticmethod
    def temperature_interpolation(inter_position):
        return inter_position

    def find_xs(self, temperature_matrix_public, temperature_matrix, channel_position, channel_index, xc_index, inv_radius, crash_end):
        """ -----------------------------------------
            version: 0.9
            desc: t => temperature
                  x = r^2
            :param temperature_matrix_public:
            :param temperature_matrix:
            :param channel_position:
            :param channel_index:
            :param xc:
            :return:
        ----------------------------------------- """

        """ convert ndarray to list """
        channel_index = [v for v in channel_index]

        t_center_precrash_norm = 1
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

    def find_center(self, temperature_matrix, channel_position, channel_index):

        xc = self.center

        channel_position_diff = [abs(x - xc) for x in channel_position]
        diff_min = min(channel_position_diff)
        xc_channel_order = channel_position_diff.index(diff_min)

        xc_index = channel_index[xc_channel_order]

        return xc_index

    def integrate_eiler(self, x2, x1):
        inter_position = x2 - x1
        temperature = self.temperature_interpolation(inter_position)

        input_function = (1 - temperature) / (temperature - 1)

        return 1
