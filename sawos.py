import controller as dt
import matplotlib.pyplot as plt

data = dt.dataPreparing()
data.getData(discharge=25, channel=55, source='real', filterType='multiple', windowW=81)

print(data.winListNames)

fig, ax = plt.subplots()
ax.plot(data.t, data.temperature['boxcar'], label="boxcar")
ax.plot(data.t, data.temperature['triang'], label="triang")
ax.plot(data.t, data.temperature['blackman'], label="blackman")

ax.set(xlabel='time (s)', ylabel='T (eV or a.u.)',
       title='JET tokamat temperature evolution, Raw data, 55 channel, 25 discharge')
ax.grid()

plt.legend()
plt.show()
