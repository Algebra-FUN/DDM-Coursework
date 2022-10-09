from matplotlib import pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(20, 20)
theta = np.linspace(0,2*np.pi,36)
rho = 1-np.sin(theta)
x = rho*np.cos(theta)
y = rho*np.sin(theta)
axes[0][0].plot(x,y,lw=3)
axes[0][0].set_title('fat heart')
axes[0][0].set_aspect('equal', adjustable='box')
x = np.linspace(-2,2,100)
y1 = np.sqrt(1-(np.abs(x)-1)**2)
y2 = np.arccos(1-np.abs(x))-np.pi
axes[0][1].plot(x,y1,c='r')
axes[0][1].plot(x,y2,c='r')
axes[0][1].set_title('better heart')
axes[1][0].plot(x,y1,c='r')
axes[1][0].plot(x,y2,c='r')
axes[1][0].set_axis_off()
axes[1][0].text(0,-.5,"better heart",fontsize=20,color="blue",horizontalalignment='center', verticalalignment='center')
axes[1][1].set_axis_off()
ax = fig.add_subplot(224,projection='polar')
ax.plot(theta,rho)
ax.text(0.5,0.5,"better heart",fontsize=20,color="blue",horizontalalignment='center', verticalalignment='center')
plt.show()