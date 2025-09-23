import time
from tango import AttrQuality, AttrWriteType, AttrDataFormat, DevState, Attr, SpectrumAttr, ImageAttr
from tango import CmdArgType, UserDefaultAttrProp, AttributeInfoEx
from tango.server import Device, attribute, command, DeviceMeta
from tango.server import class_property, device_property, run
import os
import paho.mqtt.client as mqtt
import json
from json import JSONDecodeError
import ast

class Mqtt(Device, metaclass=DeviceMeta):
    pass

    host = device_property(dtype=str, default_value="127.0.0.1")
    port = device_property(dtype=int, default_value=1883)
    username = device_property(dtype=str, default_value="")
    password = device_property(dtype=str, default_value="")
    init_subscribe = device_property(dtype=str, default_value="")
    init_dynamic_attributes = device_property(dtype=str, default_value="")
    tls_mode = device_property(dtype=bool, default_value="none")
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
            self.reconnect()

    def on_message(self, client, userdata, msg):
        value = msg.payload
        name = msg.topic
        self.info_stream("Received message: " + name + " " + str(value))
        if not name in self.dynamicAttributes:
            self.add_dynamic_attribute(name)
        if(self.dynamicAttributes[name] != value):
            self.dynamicAttributes[name] = value
            self.push_change_event(name, self.stringValueToTypeValue(name, value))

    @command(dtype_in=str)
    def add_dynamic_attribute(self, topic, 
            variable_type_name="DevString", min_value="", max_value="",
            unit="", write_type_name="READ_WRITE", label="", min_alarm="", max_alarm="",
            min_warning="", max_warning="", data_format_name=""):
        if topic == "": return
        variableType = self.stringValueToVarType(variable_type_name)
        writeType = self.stringValueToWriteType(write_type_name)
        dataFormat = self.stringValueToFormatType(data_format_name)
        prop = UserDefaultAttrProp()
        if(min_value != "" and min_value != max_value): prop.set_min_value(min_value)
        if(max_value != "" and min_value != max_value): prop.set_max_value(max_value)
        if(unit != ""):  prop.set_unit(unit)
        if(label != ""): prop.set_label(label)
        if(min_alarm != ""): prop.set_min_alarm(min_alarm)
        if(max_alarm != ""): prop.set_max_alarm(max_alarm)
        if(min_warning != ""): prop.set_min_warning(min_warning)
        if(max_warning != ""): prop.set_max_warning(max_warning)
        if dataFormat == AttrDataFormat.SCALAR:
            attr = Attr(topic, variableType, writeType)
        if dataFormat == AttrDataFormat.SPECTRUM:
            attr = SpectrumAttr(topic, variableType, writeType, 256)
        if dataFormat == AttrDataFormat.IMAGE:
            attr = ImageAttr(topic, variableType, writeType, 256, 256)
        attr.set_default_properties(prop)
        self.add_attribute(attr, r_meth=self.read_dynamic_attr, w_meth=self.write_dynamic_attr)
        self.dynamicAttributes[topic] = ""

    def stringValueToVarType(self, variable_type_name) -> CmdArgType:
        mapping = {
            "DevBoolean": CmdArgType.DevBoolean,
            "DevLong": CmdArgType.DevLong,
            "DevDouble": CmdArgType.DevDouble,
            "DevFloat": CmdArgType.DevFloat,
            "DevString": CmdArgType.DevString
        }
        if variable_type_name not in mapping:
            raise Exception("given variable_type '" + variable_type + "' unsupported, supported are:  " + ", ".join(mapping.keys()))
        return mapping[variable_type_name]

    def stringValueToWriteType(self, write_type_name) -> AttrWriteType:
        mapping = {
            "READ": AttrWriteType.READ,
            "WRITE": AttrWriteType.WRITE,
            "READ_WRITE": AttrWriteType.READ_WRITE,
            "READ_WITH_WRITE": AttrWriteType.READ_WITH_WRITE
        }
        if write_type_name not in mapping:
            raise Exception("given write_type '" + write_type_name + "' unsupported, supported are:  " + ", ".join(mapping.keys()))
        return mapping[write_type_name]

    def stringValueToFormatType(self, format_type_name) -> AttrDataFormat:
        mapping = {
            "SCALAR": AttrDataFormat.SCALAR,
            "SPECTRUM": AttrDataFormat.SPECTRUM,
            "IMAGE": AttrDataFormat.IMAGE,
        }
        if format_type_name not in mapping:
            return AttrDataFormat.SCALAR
        return mapping[format_type_name]

    def stringValueToTypeValue(self, name, val):
        attr = self.get_device_attr().get_attr_by_name(name)
        type = attr.get_data_type()
        data_format = attr.get_data_format()
        if isinstance(val, bytes): val = val.decode()
        if(data_format != AttrDataFormat.SCALAR):
            if val in ('', None): return []
            return ast.literal_eval(val)
        if(type == CmdArgType.DevBoolean):
            if str(val).lower() == "false": return False
            if str(val).lower() == "true": return True
            return bool(int(self.stringValueToFloat(val)))
        if(type == CmdArgType.DevLong):
            return int(self.stringValueToFloat(val))
        if(type == CmdArgType.DevDouble or type == CmdArgType.DevFloat):
            return self.stringValueToFloat(val)
        return val

    def stringValueToFloat(self, val):
        return float(val) if val not in ('', None) else 0

    def read_dynamic_attr(self, attr):
        name = attr.get_name()
        value = self.dynamicAttributes[name]
        self.debug_stream("read value " + str(name) + ": " + str(value))
        attr.set_value(self.stringValueToTypeValue(name, value))

    def write_dynamic_attr(self, attr):
        value = str(attr.get_write_value())
        name = attr.get_name()
        self.dynamicAttributes[name] = value
        self.publish([name, self.dynamicAttributes[name]])

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
        if self.tls_mode == "tls":
            self.client.tls_set()
        if self.username != "" and self.password != "":
            self.client.username_pw_set(self.username, self.password)
        self.info_stream("Connecting to " + str(self.host) + ":" + str(self.port))
        if self.init_dynamic_attributes != "":
            try:
                attributes = json.loads(self.init_dynamic_attributes)
                for attributeData in attributes:
                    # self.info_stream(f"Init dynamic attribute: {attributeData}") # not working for special characters
                    self.add_dynamic_attribute(
                        str(attributeData["name"]), 
                        str(attributeData.get("data_type", "")), 
                        str(attributeData.get("min_value", "")),
                        str(attributeData.get("max_value", "")),
                        str(attributeData.get("unit", "")), 
                        str(attributeData.get("write_type", "")), 
                        str(attributeData.get("label", "")),
                        str(attributeData.get("min_alarm", "")),
                        str(attributeData.get("max_alarm", "")),
                        str(attributeData.get("min_warning", "")),
                        str(attributeData.get("max_warning", "")),
                        str(attributeData.get("data_format", "")))
            except JSONDecodeError as e:
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
