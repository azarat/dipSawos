import controller as dt
import matplotlib.pyplot as plt

data = dt.dataPreparing()
data.getData(25, 55, 1)

# print((data.temperature))

fig, ax = plt.subplots()
ax.plot(data.t, data.temperature)

ax.set(xlabel='time (s)', ylabel='T (eV or a.u.)',
       title='JET tokamat temperature evolution, Raw data, 55 channel, 25 discharge')
ax.grid()

plt.show()
