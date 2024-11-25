
import cv2
import numpy as np
from matplotlib import pyplot as plt


if False:
	fig, ax = plt.subplots(figsize=(5, 2.7))

	t = np.arange(0.0, 5.0, 0.01)
	s = np.cos(2 * np.pi * t)
	ax.plot(t, s, lw=2)

	ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
				arrowprops=dict(facecolor='black', shrink=0.05))

	ax.set_ylim(-2, 2)
	plt.show()


if False:
	fig, ax = plt.subplots()             # Create a figure containing a single Axes.
	ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the Axes.
	plt.show() 
 
if False:
	data1, data2, data3, data4 = np.random.randn(4, 100)
	fig, ax = plt.subplots(figsize=(5, 2.7))
	x = np.arange(len(data1))
	ax.plot(x, np.cumsum(data1), color='blue', linewidth=3, linestyle='--')
	l, = ax.plot(x, np.cumsum(data2), color='orange', linewidth=2)
	l.set_linestyle(':')
	plt.show() 

if False:
	data1, data2, data3, data4 = np.random.randn(4, 100)
	print(np.mean(data1))
	fig, ax = plt.subplots()
	ax.scatter(data1, data2, s=50, facecolor='C0', edgecolor='k')
	plt.show() 

if True:
	data1, data2, data3, data4 = np.random.randn(4, 30)
	fig, ax = plt.subplots(figsize=(5, 2.7))
	ax.plot(data1, 'o', label='data1')
	ax.plot(data2, 'd', label='data2')
	ax.plot(data3, 'v', label='data3')
	ax.plot(data4, 's', label='data4')
	ax.legend()
	plt.show()
