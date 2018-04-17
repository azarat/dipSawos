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

ax.set(xlabel='time (s)', ylabel='T (eV or a.u.)',
       title='JET tokamat temperature evolution, Raw data, 55 channel, 25 discharge')
ax.grid()

plt.legend()
plt.show()
