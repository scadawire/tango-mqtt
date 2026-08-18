import time
from tango import AttrQuality, AttrWriteType, AttrDataFormat, DevState, Attr, SpectrumAttr, ImageAttr
from tango import CmdArgType, UserDefaultAttrProp, AttributeInfoEx
from tango.server import Device, attribute, command, DeviceMeta
from tango.server import class_property, device_property, run
import os
import paho.mqtt.client as mqtt
import json
from json import JSONDecodeError
import datetime

class Mqtt(Device, metaclass=DeviceMeta):

    host = device_property(dtype=str, default_value="127.0.0.1")
    port = device_property(dtype=int, default_value=1883)
    username = device_property(dtype=str, default_value="")
    password = device_property(dtype=str, default_value="")
    client_id = device_property(dtype=str, default_value="")
    clean_session = device_property(dtype=bool, default_value=True)
    protocol = device_property(dtype=str, default_value="3.1.1")
    keepalive = device_property(dtype=int, default_value=60)
    init_subscribe = device_property(dtype=str, default_value="")
    init_dynamic_attributes = device_property(dtype=str, default_value="")
    default_qos = device_property(dtype=int, default_value=0)
    default_retain = device_property(dtype=bool, default_value=False)
    will_topic = device_property(dtype=str, default_value="")
    will_payload = device_property(dtype=str, default_value="offline")
    will_qos = device_property(dtype=int, default_value=0)
    will_retain = device_property(dtype=bool, default_value=True)
    birth_topic = device_property(dtype=str, default_value="")
    birth_payload = device_property(dtype=str, default_value="online")
    tls_mode = device_property(dtype=str, default_value="none")
    tls_ca_certs = device_property(dtype=str, default_value="")
    tls_certfile = device_property(dtype=str, default_value="")
    tls_keyfile = device_property(dtype=str, default_value="")
    tls_insecure = device_property(dtype=bool, default_value=False)
    reconnect_delay_min = device_property(dtype=int, default_value=1)
    reconnect_delay_max = device_property(dtype=int, default_value=60)

    # per attribute state, keyed by attribute name: topic, modifier, qos, retain and the cached value.
    # topicAttributes is the reverse lookup, one topic to the attributes fed from it - several
    # attributes may share a topic when each picks a different field out of its json payload.
    # Both are re-created per instance in init_device; the class level ones are only defaults, a
    # device server hosting two Mqtt devices must not have them share one dict.
    dynamicAttributes = {}
    topicAttributes = {}
    _last_msg_at = "none"
    _connected = False
    _refused = False

    @attribute(dtype=str)
    def time(self):
        return str(datetime.datetime.now())

    @attribute(dtype=str)
    def last_msg_at(self):
        return self._last_msg_at

    @attribute(dtype=bool)
    def connected(self):
        return self._connected

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            # a refused connack does not fix itself: paho keeps retrying and keeps being refused until
            # the configuration changes, so report the reason instead of announcing the device as ON
            self._connected = False
            self._refused = True
            self.set_state(DevState.FAULT)
            self.set_status("Broker refused the connection: " + self.connect_result_text(rc))
            self.error_stream("Connection refused: %s", self.connect_result_text(rc))
            return
        self._connected = True
        self._refused = False
        self.info_stream("Connected with result code %s", rc)
        self.set_state(DevState.ON)
        self.set_status("Connected to " + str(self.host) + ":" + str(self.port))
        for topic in list(self.topicAttributes):
            self.subscribe(topic)
        birth_topic = self.birth_topic if self.birth_topic != "" else self.will_topic
        if birth_topic != "":
            # counterpart of the last will: without it the retained will payload stays the last word on
            # that topic and the device reads as offline forever after its first disconnect
            self.publish_value(birth_topic, self.birth_payload, self.will_qos, self.will_retain)

    def on_disconnect(self, client, userdata, rc, properties=None):
        self._connected = False
        if self._refused:
            # a refused connack is immediately followed by the socket closing, so this callback runs
            # right after on_connect already recorded why - leave its diagnosis in place instead of
            # replacing it with a generic "reconnecting"
            return
        if rc == 0:
            self.info_stream("Disconnected on request")
            self.set_status("Disconnected")
            return
        # the network thread started by loop_start reconnects on its own, honouring the backoff set
        # through reconnect_delay_set. The hand rolled reconnect that used to sit here raced with it
        # and reconnected without any delay at all.
        self.warn_stream("Unexpected disconnection (%s), waiting for automatic reconnect", rc)
        self.set_state(DevState.UNKNOWN)
        self.set_status("Connection lost, reconnecting")

    def on_message(self, client, userdata, msg):
        self._last_msg_at = str(datetime.datetime.now())
        self.debug_stream("Received message: %s %s", msg.topic, msg.payload)
        if msg.topic not in self.topicAttributes:
            # a concrete topic arriving through a wildcard subscription: give it an attribute of its
            # own but do not subscribe again, the wildcard already delivers it and an overlapping
            # subscription is allowed to deliver every message twice
            self.add_dynamic_attribute(msg.topic, subscribe_now=False)
        for name in self.topicAttributes.get(msg.topic, []):
            try:
                self.update_attribute_value(name, msg.payload)
            except Exception as e:
                # one attribute whose modifier does not match this payload must not stop the others
                self.error_stream("Failed to process message for %s: %s", name, str(e))

    def update_attribute_value(self, name, payload):
        config = self.dynamicAttributes[name]
        value = self.apply_modifier(payload, config["modifier"])
        if config["value"] == value:
            return
        config["value"] = value
        self.push_change_event(name, self.stringValueToTypeValue(name, value))

    @command(dtype_in=str)
    def add_dynamic_attribute(self, name,
            variable_type_name="DevString", min_value="", max_value="",
            unit="", write_type_name="READ_WRITE", label="", min_alarm="", max_alarm="",
            min_warning="", max_warning="", data_format_name="",
            topic="", modifier="", qos="", retain="", subscribe_now=True):
        if name == "":
            return
        if name in self.dynamicAttributes:
            self.info_stream("Dynamic attribute already exists: %s", name)
            return
        if topic == "":
            # the attribute name is the topic unless the descriptor separates the two, which it has to
            # as soon as a modifier splits one topic into several attributes
            topic = name
        variableType = self.stringValueToVarType(variable_type_name)
        writeType = self.stringValueToWriteType(write_type_name)
        dataFormat = self.stringValueToFormatType(data_format_name)
        qosValue = self.stringValueToQos(qos)
        retainValue = self.stringValueToBool(retain, self.default_retain)
        prop = UserDefaultAttrProp()
        if min_value != "" and min_value != max_value:
            prop.set_min_value(min_value)
        if max_value != "" and min_value != max_value:
            prop.set_max_value(max_value)
        if unit != "":
            prop.set_unit(unit)
        if label != "":
            prop.set_label(label)
        if min_alarm != "":
            prop.set_min_alarm(min_alarm)
        if max_alarm != "":
            prop.set_max_alarm(max_alarm)
        if min_warning != "":
            prop.set_min_warning(min_warning)
        if max_warning != "":
            prop.set_max_warning(max_warning)
        if dataFormat == AttrDataFormat.SCALAR:
            attr = Attr(name, variableType, writeType)
        elif dataFormat == AttrDataFormat.SPECTRUM:
            attr = SpectrumAttr(name, variableType, writeType, 256)
        elif dataFormat == AttrDataFormat.IMAGE:
            attr = ImageAttr(name, variableType, writeType, 256, 256)
        else:
            attr = Attr(name, variableType, writeType)
        attr.set_default_properties(prop)
        self.add_attribute(attr, r_meth=self.read_dynamic_attr, w_meth=self.write_dynamic_attr)
        self.set_change_event(name, True, False)
        self.dynamicAttributes[name] = {
            "topic": topic,
            "modifier": modifier,
            "qos": qosValue,
            "retain": retainValue,
            "value": "",
        }
        self.topicAttributes.setdefault(topic, []).append(name)
        if subscribe_now and self._connected:
            # attributes added at runtime through this command would otherwise stay silent until the
            # next reconnect, since on_connect is the only other place that subscribes
            self.subscribe(topic)

    def stringValueToVarType(self, variable_type_name) -> CmdArgType:
        mapping = {
            "DevBoolean": CmdArgType.DevBoolean,
            "DevLong": CmdArgType.DevLong,
            "DevDouble": CmdArgType.DevDouble,
            "DevFloat": CmdArgType.DevFloat,
            "DevString": CmdArgType.DevString
        }
        if variable_type_name not in mapping:
            raise Exception(
                "given variable_type '" + variable_type_name +
                "' unsupported, supported are: " + ", ".join(mapping.keys())
            )
        return mapping[variable_type_name]

    def stringValueToWriteType(self, write_type_name) -> AttrWriteType:
        # READ_WITH_WRITE is deliberately not offered: tango only accepts it for an attribute that
        # names an associated write attribute, and every attribute here is built as a plain
        # Attr(topic, type, writeType) without one. Constructing it anyway does not fail that single
        # attribute, it aborts init_device with "Associated attribute not defined" and takes the
        # whole device server down, so it is rejected up front with a message naming the real options.
        mapping = {
            "READ": AttrWriteType.READ,
            "WRITE": AttrWriteType.WRITE,
            "READ_WRITE": AttrWriteType.READ_WRITE
        }
        if write_type_name not in mapping:
            raise Exception(
                "given write_type '" + write_type_name +
                "' unsupported, supported are: " + ", ".join(mapping.keys())
            )
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

    def stringValueToProtocol(self, protocol_name):
        mapping = {
            "3.1": mqtt.MQTTv31,
            "3.1.1": mqtt.MQTTv311,
            "5": mqtt.MQTTv5,
            "5.0": mqtt.MQTTv5,
        }
        if str(protocol_name) not in mapping:
            raise Exception(
                "given protocol '" + str(protocol_name) +
                "' unsupported, supported are: " + ", ".join(mapping.keys())
            )
        return mapping[str(protocol_name)]

    def stringValueToQos(self, qos) -> int:
        # an omitted qos falls back to the device wide default_qos rather than to a hardcoded 0
        if qos in ("", None):
            return self.default_qos
        value = int(qos)
        if value not in (0, 1, 2):
            raise Exception("given qos '" + str(qos) + "' unsupported, supported are: 0, 1, 2")
        return value

    def stringValueToBool(self, value, default) -> bool:
        # descriptors reach this both as real json booleans and as the strings a tango property carries
        if value in ("", None):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def connect_result_text(self, rc):
        # mqtt 5 hands over a ReasonCodes object that already renders its own text, mqtt 3 a plain int
        if isinstance(rc, int):
            return mqtt.connack_string(rc)
        return str(rc)

    def apply_modifier(self, payload, modifier):
        """Pick a single field out of a json payload. The modifier is a dotted path with list indexes
           allowed - "temp", "sensor.temp", "values.0" - and an empty modifier passes the payload
           through untouched. This is what lets several attributes share one topic, and why an
           attribute name and its topic are separate keys in a descriptor."""
        if isinstance(payload, bytes):
            try:
                payload = payload.decode()
            except UnicodeDecodeError:
                # a payload that is not text carries no json field, hand it on unchanged
                return payload
        if modifier == "":
            return payload
        data = json.loads(payload)
        for part in modifier.split("."):
            if isinstance(data, list):
                data = data[int(part)]
            else:
                data = data[part]
        if data is None:
            # empty reads back as 0 for the numeric types, which beats the string "None"
            return ""
        if isinstance(data, bool):
            # json spelling, so a DevString attribute reads back what the payload actually said
            return "true" if data else "false"
        if isinstance(data, (dict, list)):
            # a spectrum or image field stays json so the array conversion below can parse it again
            return json.dumps(data)
        return str(data)

    def _cast_element(self, val, data_type):
        if data_type == CmdArgType.DevBoolean:
            return bool(val)
        if data_type == CmdArgType.DevLong:
            return int(val)
        if data_type in (CmdArgType.DevDouble, CmdArgType.DevFloat):
            return float(val)
        return val

    def _cast_array(self, data, data_type):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return [[self._cast_element(e, data_type) for e in row] for row in data]
        return [self._cast_element(e, data_type) for e in data]

    def stringValueToTypeValue(self, name, val):
        attr = self.get_device_attr().get_attr_by_name(name)
        data_type = attr.get_data_type()
        data_format = attr.get_data_format()
        if isinstance(val, bytes):
            val = val.decode()
        if data_format != AttrDataFormat.SCALAR:
            if val in ('', None):
                return []
            parsed = json.loads(val) if isinstance(val, str) else val
            return self._cast_array(parsed, data_type)
        if data_type == CmdArgType.DevBoolean:
            if str(val).lower() == "false":
                return False
            if str(val).lower() == "true":
                return True
            return bool(int(self.stringValueToFloat(val)))
        if data_type == CmdArgType.DevLong:
            return int(self.stringValueToFloat(val))
        if data_type in (CmdArgType.DevDouble, CmdArgType.DevFloat):
            return self.stringValueToFloat(val)
        return val

    def stringValueToFloat(self, val):
        return float(val) if val not in ('', None) else 0.0

    def read_dynamic_attr(self, attr):
        name = attr.get_name()
        value = self.dynamicAttributes[name]["value"]
        self.debug_stream("read value %s: %s", name, value)
        attr.set_value(self.stringValueToTypeValue(name, value))

    def write_dynamic_attr(self, attr):
        name = attr.get_name()
        config = self.dynamicAttributes[name]
        if config["modifier"] != "":
            # publishing back would mean rebuilding the whole json document of the topic, and every
            # other field in it belongs to the publisher - refuse rather than clobber them
            raise Exception(
                "attribute '" + name + "' reads field '" + config["modifier"] + "' out of topic '" +
                config["topic"] + "' and cannot be written, declare it with write_type READ"
            )
        value = attr.get_write_value()
        attr_info = self.get_device_attr().get_attr_by_name(name)
        if attr_info.get_data_format() != AttrDataFormat.SCALAR:
            # a numeric spectrum or image arrives as a numpy array, a DevString one as a plain list
            # (of lists for an image), and only the former carries tolist() - calling it on the
            # string case raised AttributeError and made every write of one fail
            value = json.dumps(value.tolist() if hasattr(value, "tolist") else list(value))
        else:
            value = str(value)
        config["value"] = value
        self.publish_value(config["topic"], value, config["qos"], config["retain"])
        self.push_change_event(name, self.stringValueToTypeValue(name, value))

    @command(dtype_in=str)
    def subscribe(self, topic):
        # one subscription per topic, so attributes sharing it agree on the strongest qos any of them
        # asked for - a lower one cannot be honoured for the others once the messages arrive
        qos = self.default_qos
        for name in self.topicAttributes.get(topic, []):
            qos = max(qos, self.dynamicAttributes[name]["qos"])
        self.info_stream("Subscribe to topic %s with qos %s", topic, qos)
        self.client.subscribe(topic, qos)

    @command(dtype_in=[str])
    def publish(self, args):
        """[topic, value], optionally followed by qos and retain."""
        topic = args[0]
        value = args[1]
        qos = self.stringValueToQos(args[2] if len(args) > 2 else "")
        retain = self.stringValueToBool(args[3] if len(args) > 3 else "", self.default_retain)
        self.publish_value(topic, value, qos, retain)

    def publish_value(self, topic, value, qos, retain):
        self.debug_stream("Publish topic %s: %s (qos %s, retain %s)", topic, value, qos, retain)
        self.client.publish(topic, value, qos=qos, retain=retain)

    def build_client(self):
        protocol = self.stringValueToProtocol(self.protocol)
        if not self.clean_session and self.client_id == "":
            raise Exception("clean_session=False needs a non empty client_id, a broker cannot "
                            "restore a session it has no name for")
        if protocol == mqtt.MQTTv5:
            # paho rejects clean_session for mqtt 5, there the equivalent is clean_start on connect
            client = mqtt.Client(client_id=self.client_id, protocol=protocol)
        else:
            client = mqtt.Client(client_id=self.client_id, clean_session=self.clean_session,
                                 protocol=protocol)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.on_disconnect = self.on_disconnect
        client.reconnect_delay_set(min_delay=self.reconnect_delay_min,
                                   max_delay=self.reconnect_delay_max)
        if self.tls_mode == "tls":
            client.tls_set(
                ca_certs=self.tls_ca_certs if self.tls_ca_certs != "" else None,
                certfile=self.tls_certfile if self.tls_certfile != "" else None,
                keyfile=self.tls_keyfile if self.tls_keyfile != "" else None,
            )
            if self.tls_insecure:
                client.tls_insecure_set(True)
        if self.username != "":
            # an empty password stays legal, brokers authenticating on the username alone accept it
            client.username_pw_set(self.username, self.password if self.password != "" else None)
        if self.will_topic != "":
            client.will_set(self.will_topic, self.will_payload, self.will_qos, self.will_retain)
        return client

    def teardown_client(self):
        client = self.__dict__.get("client")
        if client is None:
            return
        self.info_stream("Dropping the previous broker connection")
        try:
            client.loop_stop()
            client.disconnect()
        except Exception as e:
            self.warn_stream("Disconnect of the previous client failed: %s", str(e))

    def reconnect(self):
        # connect_async plus loop_start brings the network thread up without blocking and keeps
        # retrying on the configured backoff, so a broker that is not up yet leaves the device server
        # running and reporting FAULT instead of aborting init_device
        protocol = self.stringValueToProtocol(self.protocol)
        if protocol == mqtt.MQTTv5:
            self.client.connect_async(self.host, self.port, self.keepalive,
                                      clean_start=bool(self.clean_session))
        else:
            self.client.connect_async(self.host, self.port, self.keepalive)
        self.client.loop_start()
        self.info_stream("Connection attempted, waiting for connection result")

    def init_device(self):
        self.get_device_properties(self.get_device_class())
        self.teardown_client()
        self.set_state(DevState.INIT)
        self.set_status("Connecting to " + str(self.host) + ":" + str(self.port))
        self._connected = False
        # init_device runs again on every Init command, but the attributes added on the first pass stay
        # registered with tango, so the per instance state is created once and then reused - resetting
        # it would make add_dynamic_attribute add attributes tango already knows
        if "dynamicAttributes" not in self.__dict__:
            self.dynamicAttributes = {}
            self.topicAttributes = {}
        self.client = self.build_client()
        self.info_stream("Connecting to %s:%s", self.host, self.port)
        if self.init_dynamic_attributes != "":
            try:
                attributes = json.loads(self.init_dynamic_attributes)
                for attributeData in attributes:
                    modifier = str(attributeData.get("modifier", ""))
                    # a single field of a shared json payload cannot be published back on its own, so a
                    # descriptor naming one but no write_type gets READ instead of the READ_WRITE default
                    defaultWriteType = "READ" if modifier != "" else "READ_WRITE"
                    self.add_dynamic_attribute(
                        str(attributeData["name"]),
                        # every key but name is optional, so an absent one has to fall back to the value
                        # the signature already defaults to - it used to reach the mappers as "" and
                        # abort init_device with "given variable_type '' unsupported"
                        variable_type_name=str(attributeData.get("data_type", "DevString")),
                        min_value=str(attributeData.get("min_value", "")),
                        max_value=str(attributeData.get("max_value", "")),
                        unit=str(attributeData.get("unit", "")),
                        write_type_name=str(attributeData.get("write_type", defaultWriteType)),
                        label=str(attributeData.get("label", "")),
                        min_alarm=str(attributeData.get("min_alarm", "")),
                        max_alarm=str(attributeData.get("max_alarm", "")),
                        min_warning=str(attributeData.get("min_warning", "")),
                        max_warning=str(attributeData.get("max_warning", "")),
                        data_format_name=str(attributeData.get("data_format", "")),
                        topic=str(attributeData.get("topic", "")),
                        modifier=modifier,
                        qos=attributeData.get("qos", ""),
                        retain=attributeData.get("retain", ""),
                    )
            except JSONDecodeError as e:
                attributes = self.init_dynamic_attributes.split(",")
                for attribute in attributes:
                    self.info_stream("Init dynamic attribute: %s", str(attribute.strip()))
                    self.add_dynamic_attribute(attribute.strip())

        if self.init_subscribe != "":
            init_subscribes = self.init_subscribe.split(",")
            for init_sub in init_subscribes:
                topic = init_sub.strip()
                self.info_stream("Init subscribe: %s", topic)
                self.add_dynamic_attribute(topic)

        self.reconnect()


if __name__ == "__main__":
    deviceServerName = os.getenv("DEVICE_SERVER_NAME", "Mqtt")
    run({deviceServerName: Mqtt})
