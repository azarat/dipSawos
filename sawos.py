import controller as dt
import matplotlib.pyplot as plt

data = dt.dataPreparing()
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
# plt.show()
fig.savefig('results/filters/d25_c55_w81.png')
