import time
from tango import AttrQuality, AttrWriteType, DispLevel, DevState, Attr, CmdArgType
from tango.server import Device, attribute, command, DeviceMeta
from tango.server import class_property, device_property
from tango.server import run
import os
import paho.mqtt.client as mqtt

class Mqtt(Device, metaclass=DeviceMeta):
    pass

    host = device_property(dtype=str, default_value="127.0.0.1")
    port = device_property(dtype=int, default_value=1883)
    username = device_property(dtype=str, default_value="")
    password = device_property(dtype=str, default_value="")
    init_subscribe = device_property(dtype=str, default_value="")
    init_dynamic_attributes = device_property(dtype=str, default_value="")
    client = mqtt.Client()
    dynamicAttributes = {}

    @attribute
    def time(self):
        return time.time()

    def on_connect(self, client, userdata, flags, rc):
        self.info_stream("Connected with result code " + str(rc))
        self.set_state(DevState.ON)
        for key in self.dynamicAttributes:
            self.subscribe(key)

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self.warn_stream("Unexpected disconnection. trying reconnect")        
            self.set_state(DevState.UNKNOWN)
            self.reconnect();

    def on_message(self, client, userdata, msg):
        self.info_stream("Received message: " + msg.topic+" "+str(msg.payload))
        if not msg.topic in self.dynamicAttributes:
            self.add_dynamic_attribute(msg.topic)
        self.dynamicAttributes[msg.topic] = msg.payload
        self.push_change_event(msg.topic, msg.payload)

    @command(dtype_in=str)
    def add_dynamic_attribute(self, topic):
        if topic == "": return
        attr = Attr(topic, CmdArgType.DevString, AttrWriteType.READ_WRITE)
        self.add_attribute(attr, r_meth=self.read_dynamic_attr, w_meth=self.write_dynamic_attr)
        self.dynamicAttributes[topic] = ""

    def read_dynamic_attr(self, attr):
        attr.set_value(self.dynamicAttributes[attr.get_name()])

    def write_dynamic_attr(self, attr):
        self.dynamicAttributes[attr.get_name()] = attr.get_write_value()
        self.publish([attr.get_name(), self.dynamicAttributes[attr.get_name()]])

    @command(dtype_in=str)
    def subscribe(self, topic):
        self.info_stream("Subscribe to topic " + str(topic))
        self.client.subscribe(topic)

    @command(dtype_in=[str])
    def publish(self, args):
        topic, value = args
        self.info_stream("Publish topic " + str(topic) + ": " + str(value))
        self.client.publish(topic, value)

    def reconnect(self):
        self.client.connect(self.host, self.port, 60)
        self.client.loop_start()
        self.info_stream("Connection attempted, waiting for connection result")
        
    def init_device(self):
        self.set_state(DevState.INIT)
        self.get_device_properties(self.get_device_class())
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        if self.username != "" and self.password != "":
            # 2024-01-16 tls enabling not required/possible for localhost connection with authentication enabled -> disable
            # self.client.tls_set()  # is necessary for authentication?
            self.client.username_pw_set(self.username, self.password)
        self.info_stream("Connecting to " + str(self.host) + ":" + str(self.port))
        if self.init_dynamic_attributes != "":
            attributes = self.init_dynamic_attributes.split(",")
            for attribute in attributes:
                self.info_stream("Init dynamic attribute: " + str(attribute.strip()))
                self.add_dynamic_attribute(attribute.strip())
        if self.init_subscribe != "":
            init_subscribes = self.init_subscribe.split(",")
            for init_subscribe in init_subscribes:
                self.info_stream("Init subscribe: " + str(init_subscribe.strip()))
                self.add_dynamic_attribute(init_subscribe.strip())
        self.reconnect()

if __name__ == "__main__":
    deviceServerName = os.getenv("DEVICE_SERVER_NAME")
    run({deviceServerName: Mqtt})
