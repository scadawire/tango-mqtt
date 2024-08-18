# tango-mqtt

scadawire/tango-controls integration to Mqtt servers/devices

This device driver integrates to IOT devices, see also https://mqtt.org/.

# Structure

The integration makes use of the tango python binding to implement device driver functionality.
See also https://tango-controls.readthedocs.io/en/latest/development/device-api/python/index.html

The mqtt specific functionality is covered by the mqtt paho python package.
See also https://pypi.org/project/paho-mqtt/

# Requirements

The device server supports 5.0, 3.1.1, and 3.1 of the MQTT protocol.
The integration allows for authenticated and unauthenticated connections.
