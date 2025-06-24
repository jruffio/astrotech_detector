import PySpin
import numpy as np
from astropy.io import fits
import os

def acquire_images(tint, Ncoadd, outdir=None,prefix="image"):
    """
    Acquire a series of images from a FLIR/Spinnaker camera using PySpin.

    Parameters
    ----------
    tint : float or int
        Exposure time in microseconds (us) to set for the camera.
    Ncoadd : int
        Number of images to acquire (coadds). Each image will be acquired sequentially with the same exposure time.
    outdir : str or None
        Directory where to save the combined (mean) image as a FITS file. If None, no file is saved.
    prefix : str
        Prefix for the output FITS file name. This will be used to create the filename in the format `prefix_tint{tint}_N{Ncoadd}.fits`.

    Returns
    -------
    np.ndarray
        3D numpy array of shape (Ncoadd, height, width), where each slice along the first axis is a single acquired image.

    Raises
    ------
    RuntimeError
        If no cameras are detected or if no complete images are acquired.

    Notes
    -----
    - This function connects to the first detected camera, sets the exposure time, and acquires Ncoadd images.
    - The camera is initialized and deinitialized within the function.
    - All resources are released after acquisition.
    - Incomplete images are skipped and not included in the output array.
    - If outdir is provided, the mean image is saved as a FITS file in that directory.
    """
    # Initialize PySpin system and camera list
    # print("Initializing PySpin system...")
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    # print(f"Number of cameras detected: {num_cameras}")
    if num_cameras == 0:
        system.ReleaseInstance()
        raise RuntimeError("No cameras detected.")
    # Use the first available camera
    # print("Initializing first camera...")
    cam = cam_list.GetByIndex(0)
    cam.Init()

    # Turn off auto gain
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    # Set gain to 0 dB
    cam.Gain.SetValue(0)

    # #Set auto white balance to off
    # cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)

    # Make sure gamma is enabled
    cam.GammaEnable.SetValue(False)
    
    # After cam.Init()
    nodemap = cam.GetNodeMap()
    node = PySpin.CFloatPtr(nodemap.GetNode("DeviceTemperature"))
    if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
        temperature = node.GetValue()
        print(f"Camera temperature: {temperature:.2f} °C")
    else:
        temperature = None
        print("DeviceTemperature node not available or not readable on this camera.")

    # Set camera exposure settings
    # print("Turning off auto exposure...")
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
    cam.ExposureTime.SetValue(float(tint))
    images = []
    # Begin image acquisition
    for i in range(Ncoadd):
        cam.BeginAcquisition()
        img = cam.GetNextImage()
        if img.IsIncomplete():
            img.Release()
            continue
        data = img.GetNDArray()
        images.append(data)
        img.Release()
        cam.EndAcquisition()
    # Deinitialize and clean up
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    if not images:
        raise RuntimeError("No complete images acquired.")
    stack = np.stack(images, axis=0)
    # Save stack and mean image as FITS extensions if requested
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        mean_image = np.mean(stack, axis=0)
        fits_path = os.path.join(outdir, prefix+f"_tint{int(tint)}_coadd{Ncoadd}.fits")
        hdu_stack = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(stack, name='STACK')
        hdu2 = fits.ImageHDU(mean_image, name='MEAN')
        hdul = fits.HDUList([hdu_stack, hdu1, hdu2])
        # Add detector temperature to header
        if temperature is not None:
            for hdu in hdul:
                hdu.header['DETTEMP'] = (float(temperature), 'Detector temperature in deg C')
        hdul.writeto(fits_path, overwrite=True)
        print(f"Saved stack and mean image to {fits_path}")
    return stack
