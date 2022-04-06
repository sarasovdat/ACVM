
import matplotlib.pyplot as plt
import numpy as np

# AR plots

#################################################################################

# Calibrating alpha

# 0: (0.17331090338067812, 0.4174708742769856)
# 0.001: (0.18560619664990988, 0.4153651130591887)
# 0.01: (0.3118660122731414, 0.393334224958471)
# 0.02: (0.4183493780686673, 0.4088182267693448)
# 0.05: (0.47050859349713287, 0.39918154836861464)
# 0.08: (0.49411655707934093, 0.4141325700260536)
# 0.1: (0.5343777431711156, 0.42232459636444775)
# 0.15: (0.5138529854302103, 0.41500621989868974)
# 0.2: (0.48453445836732933, 0.4243144377924591)
# 0.5: (0.5088461755294741, 0.40835292921404837)
# 0.9: (0.4659241167088338, 0.4158446276199894)

r = [0.18560619664990988, 0.17331090338067812, 0.3118660122731414, 0.4183493780686673, 0.47050859349713287, 
     0.49411655707934093, 0.5138529854302103, 0.48453445836732933, 0.5088461755294741, 0.4659241167088338]

a = [0.4153651130591887, 0.4174708742769856, 0.393334224958471, 0.4088182267693448, 
     0.39918154836861464, 0.4141325700260536, 0.41500621989868974, 4243144377924591,
     0.40835292921404837, 0.4158446276199894]

r_best, a_best = 0.5343777431711156, 0.42232459636444775

fig = plt.figure()
plt.plot(r, a, marker = 'o', markerfacecolor = 'k', markeredgewidth = 0, linestyle = '', markersize = 5, label = 'Performance: alpha calibrations')
plt.plot(r_best, a_best, marker = 'o', markerfacecolor = 'magenta', markeredgewidth = 0, linestyle = '', markersize = 10, label = 'Performance: best alpha calibration')
x = np.linspace(0, 1, 100)
plt.plot(x, x, linestyle = 'dotted', color = 'k', linewidth = 0.5)
plt.axis('square')
plt.axis([0, 1, 0, 1])
plt.legend()
plt.title('AR Plot: Calibrating parameter alpha')
fig.axes[0].set_xlabel('Robustness')
fig.axes[0].set_ylabel('Accuracy')
fig.savefig("calibrating_alpha.png")

#################################################################################

# Calibrating sigma

# 0: (0.2855595723966548, 0.33169719145271764)
# 0.01: (0.33399089886023425, 0.3498685196066429)
# 0.25: (0.31803343930112205, 0.3462154203446678)
# 0.5: (0.4183493780686673, 0.37633662952921826)
# 0.75: (0.4751381794065713, 0.39581391148396433)
# 1.0: (0.49897843437954525, 0.38875795026185095)
# 1.5: (0.5240148840213757, 0.38702381139753)
# 1.75: (0.5343777431711156, 0.40549199623212323)
# 2: (0.5343777431711156, 0.42232459636444775)
# 2.25: (0.5088461755294741, 0.42629136760419156)
# 2.5: (0.5189090600136496, 0.42761133346802543)
# 3: (0.5240148840213757, 0.43179704896735216)
# 4: (0.4524369671617328, 0.41232769368951677)

r = [0.2855595723966548, 0.33399089886023425, 0.31803343930112205, 0.4183493780686673, 0.4751381794065713,
     0.49897843437954525, 0.5240148840213757, 0.5343777431711156,
     0.5088461755294741, 0.5189090600136496, 0.5240148840213757, 0.4524369671617328]
     
a = [0.33169719145271764, 0.3498685196066429, 0.3462154203446678, 0.37633662952921826, 0.39581391148396433,
     0.38875795026185095, 0.38702381139753, 0.40549199623212323, 
     0.42629136760419156, 0.42761133346802543, 0.43179704896735216, 0.41232769368951677]

r_best, a_best = 0.5343777431711156, 0.42232459636444775

fig = plt.figure()
plt.plot(r, a, marker = 'o', markerfacecolor = 'k', markeredgewidth = 0, linestyle = '', markersize = 5, label = 'Performance: sigma calibrations')
plt.plot(r_best, a_best, marker = 'o', markerfacecolor = 'magenta', markeredgewidth = 0, linestyle = '', markersize = 10, label = 'Performance: best sigma calibration')
x = np.linspace(0, 1, 100)
plt.plot(x, x, linestyle = 'dotted', color = 'k', linewidth = 0.5)
plt.axis('square')
plt.axis([0, 1, 0, 1])
plt.legend()
plt.title('AR Plot: Calibrating parameter sigma')
fig.axes[0].set_xlabel('Robustness')
fig.axes[0].set_ylabel('Accuracy')
fig.savefig("calibrating_sigma.png")


#################################################################################

# Calibrating increase factor

# 1: (0.4613843095136368, 0.4493149903014187)
# 1.2: (0.5343777431711156, 0.42232459636444775)
# 1.5: (0.5449455370527341, 0.3390380044115181)
# 1.75: (0.583605915962105, 0.27211592692808706)
# 2: (0.631158798070927, 0.2305799806783438)
# 2.5: (0.6010032129038911, 0.15892139550665665)

r = [0.4613843095136368, 0.5449455370527341, 0.583605915962105, 
     0.631158798070927, 0.6010032129038911]

a = [0.4493149903014187, 0.3390380044115181, 0.27211592692808706,
     0.2305799806783438, 0.15892139550665665]


r_best, a_best = 0.5343777431711156, 0.42232459636444775

fig = plt.figure()
plt.plot(r, a, marker = 'o', markerfacecolor = 'k', markeredgewidth = 0, linestyle = '', markersize = 5, label = 'Performance: increase factor calibrations')
plt.plot(r_best, a_best, marker = 'o', markerfacecolor = 'magenta', markeredgewidth = 0, linestyle = '', markersize = 10, label = 'Performance: best increase factor calibration')
x = np.linspace(0, 1, 100)
plt.plot(x, x, linestyle = 'dotted', color = 'k', linewidth = 0.5)
plt.axis('square')
plt.axis([0, 1, 0, 1])
plt.legend()
plt.title('AR Plot: Calibrating parameter increase_factor')
fig.axes[0].set_xlabel('Robustness')
fig.axes[0].set_ylabel('Accuracy')
fig.savefig("calibrating_inc.png")