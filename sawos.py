from scipy.io import loadmat
import matplotlib.pyplot as plt

sawdata = loadmat('saw_data.mat')
data = sawdata['saw_data'][0,0]['KK3JPF'][0,0]['TE55'][0,25]

t = range(0, len(data))

fig, ax = plt.subplots()
ax.plot(t, data * 27.211 * 1000)
# ax.legend()

ax.set(xlabel='time (s)', ylabel='T (eV or a.u.)',
       title='JET tokamat temperature evolution, Raw data, 55 channel, 25 discharge')
ax.grid()

plt.show()
