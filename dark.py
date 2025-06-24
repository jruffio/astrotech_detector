import numpy as np
import matplotlib.pyplot as plt
from utils import acquire_images

# Define exposure times in microseconds (us)
exposure_times = np.logspace(5, 7, num=10, dtype=int) 
Ncoadd = 5  # Number of images to coadd for each exposure time

dark_currents = []

for tint in exposure_times:
    images = acquire_images(tint, Ncoadd)
    # Compute mean signal per pixel for each image, then average over Ncoadd
    mean_signal = np.mean(images, axis=(1,2))
    dark_current = np.mean(mean_signal)
    dark_currents.append(dark_current)

dark_currents = np.array(dark_currents)

plt.figure(figsize=(8,6))
plt.plot(exposure_times/1e6, dark_currents, marker='o')
plt.xlabel('Exposure Time (s)')
plt.ylabel('Mean Dark Signal (ADU)')
plt.title('Dark Current vs Exposure Time')
plt.grid(True)
plt.tight_layout()
plt.show()
