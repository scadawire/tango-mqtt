# tango-symcon

scadawire/tango-controls integration to IP-Symcon

This device driver integrates to the German building automation software of IP-Symcon. See also https://www.symcon.de/de/

# Structure

The integration makes use of the tango python binding to implement device driver functionality.
See also https://tango-controls.readthedocs.io/en/latest/development/device-api/python/index.html

The ip symcon specific functionality is covered by the symcon python package.
See also https://github.com/thorstenMueller/symcon-python

# Requirements

The IP-Symcon Kernel version is required in version 6 and up.
