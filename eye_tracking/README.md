# Setup Tobii Eye Tracker 4C

Hardware is compatible to Ubuntu and it is possible to calibrate the device using the official (Tobii software)[https://www.tobiipro.com/product-listing/eye-tracker-manager/] but as of right now the Python library is blocked due to the lack of license (you can reproduce this error by connecting a Tobii 4C to your machine and running ```python find_eye_tracker.py```):

```
Address: tobii-ttp://IS404-100210417166
Model: IS4_Large_Peripheral
Name (It's OK if this is empty): 
Serial number: IS404-100210417166
Result: (201,)
Traceback (most recent call last):
  File "find_eye_tracker.py", line 25, in <module>
    my_eyetracker.retrieve_calibration_data()
  File "/home/ultron/Projects/TobiiPro.SDK.Python.LINUX_1.7.0.3/64/tobiiresearch/implementation/EyeTracker.py", line 486, in retrieve_calibration_data
    return interop.calibration_retrieve(self.__core_eyetracker)
  File "/home/ultron/Projects/TobiiPro.SDK.Python.LINUX_1.7.0.3/64/tobiiresearch/interop/interop.py", line 318, in calibration_retrieve
    _on_error_raise_exception(result[0])
  File "/home/ultron/Projects/TobiiPro.SDK.Python.LINUX_1.7.0.3/64/tobiiresearch/implementation/Errors.py", line 168, in _on_error_raise_exception
    raise EyeTrackerLicenseError("Insufficient license level when using a restricted feature. " + str(status))
tobiiresearch.implementation.Errors.EyeTrackerLicenseError: "Insufficient license level when using a restricted feature. 'se_insufficient_license'"
```

I'm already contacting Tobii to check the prices for the license and also looking for license-free alternatives to simply get the eye gaze coordinates.