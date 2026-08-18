"""
Full test for Mqtt.py covering all var types and data formats.

Tests scalar, spectrum (1D), and image (2D) for every supported data type
(DevBoolean, DevLong, DevFloat, DevDouble, DevString).

All conversion logic is exercised through the real Mqtt class via unbound
method calls -- no code is duplicated.

Usage:
    python test_mqtt.py
"""

import json
import sys
import traceback

import numpy as np
import paho.mqtt.client as mqtt
from tango import CmdArgType, AttrDataFormat, AttrWriteType, DevState

from Mqtt import Mqtt


# ===========================================================================
#  State carrier + mock attr registry
# ===========================================================================

class MockAttrInfo:
    """Mimics the object returned by Tango's get_attr_by_name()."""
    def __init__(self, data_type, data_format):
        self._data_type = data_type
        self._data_format = data_format

    def get_data_type(self):
        return self._data_type

    def get_data_format(self):
        return self._data_format


class MockDeviceAttr:
    def __init__(self):
        self._attrs = {}

    def register(self, name, data_type, data_format):
        self._attrs[name] = MockAttrInfo(data_type, data_format)

    def get_attr_by_name(self, name):
        return self._attrs[name]


class MockMessage:
    """Mimics the paho message handed to on_message."""
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload


class MockClient:
    """Mimics the paho client, recording what the driver asks the broker for."""
    def __init__(self):
        self.subscriptions = []
        self.published = []

    def subscribe(self, topic, qos=0):
        self.subscriptions.append((topic, qos))

    def publish(self, topic, value, qos=0, retain=False):
        self.published.append((topic, value, qos, retain))


class State:
    """Carries instance state; every method lookup falls through to Mqtt."""

    def __init__(self):
        self._device_attr = MockDeviceAttr()
        self.dynamicAttributes = {}
        self.topicAttributes = {}
        self.published = []
        self.events = []
        self.client = MockClient()
        self.state = None
        self.status = ""
        # the device properties the conversion helpers read; the real ones are device_property
        # descriptors on Mqtt, which __getattr__ deliberately will not fall through to
        self.default_qos = 0
        self.default_retain = False
        self.will_topic = ""
        self.birth_topic = ""
        self.birth_payload = "online"
        self.will_qos = 0
        self.will_retain = True
        self.host = "127.0.0.1"
        self.port = 1883
        self._connected = False
        self._refused = False
        self._last_msg_at = "none"

    def get_device_attr(self):
        return self._device_attr

    def publish_value(self, topic, value, qos, retain):
        self.published.append((topic, value, qos, retain))

    def push_change_event(self, name, value):
        # defined here rather than fallen through to Mqtt: that one resolves to the real tango
        # DeviceImpl method, which rejects a mock as its self
        self.events.append((name, value))

    def add_attribute(self, attr, r_meth=None, w_meth=None):
        # an attribute created by the driver itself is a DevString scalar unless a test registers a
        # type for it afterwards, which matches what add_dynamic_attribute defaults to
        self._device_attr.register(attr.get_name(), CmdArgType.DevString, AttrDataFormat.SCALAR)

    def set_change_event(self, name, implemented, detect):
        pass

    def set_state(self, state):
        self.state = state

    def set_status(self, status):
        self.status = status

    def debug_stream(self, message, *args):
        pass

    def info_stream(self, message, *args):
        pass

    def warn_stream(self, message, *args):
        pass

    def error_stream(self, message, *args):
        pass

    def __getattr__(self, name):
        import functools
        attr = getattr(Mqtt, name, None)
        if attr is not None and callable(attr):
            return functools.partial(attr, self)
        raise AttributeError(f"'State' has no attribute '{name}'")


# Thin helpers

def register_attr(s, name, data_type, data_format=AttrDataFormat.SCALAR,
                  topic=None, modifier="", qos=0, retain=False):
    s._device_attr.register(name, data_type, data_format)
    s.dynamicAttributes[name] = {
        "topic": topic if topic is not None else name,
        "modifier": modifier,
        "qos": qos,
        "retain": retain,
        "value": "",
    }
    s.topicAttributes.setdefault(s.dynamicAttributes[name]["topic"], []).append(name)


def convert(s, name, val):
    """Simulate MQTT payload -> typed value (calls real Mqtt.stringValueToTypeValue)."""
    return Mqtt.stringValueToTypeValue(s, name, val)


class MockWriteAttr:
    """Mimics the attribute handed to a write method by Tango."""
    def __init__(self, name, write_value):
        self._name = name
        self._write_value = write_value

    def get_name(self):
        return self._name

    def get_write_value(self):
        return self._write_value


def serialize_write(s, name, value):
    """Tango write -> MQTT string, through the real Mqtt.write_dynamic_attr. This used to repeat the
       driver's logic instead of calling it, which is why a spectrum of DevString serialising through
       a numpy only code path went unnoticed here while failing on every real write."""
    Mqtt.write_dynamic_attr(s, MockWriteAttr(name, value))
    return s.dynamicAttributes[name]["value"]


# ===========================================================================
#  Test helpers
# ===========================================================================

passed = 0
failed = 0
errors = []


def assert_equal(test_name, actual, expected, tolerance=None):
    global passed, failed
    if tolerance is not None:
        ok = abs(actual - expected) <= tolerance
    else:
        ok = (actual == expected)

    if ok:
        passed += 1
        print(f"  PASS  {test_name}")
    else:
        failed += 1
        msg = f"  FAIL  {test_name}: expected {expected!r}, got {actual!r}"
        print(msg)
        errors.append(msg)


def assert_list_equal(test_name, actual, expected, tolerance=None):
    global passed, failed
    ok = False
    if len(actual) == len(expected):
        if tolerance is not None:
            ok = all(abs(a - e) <= tolerance for a, e in zip(actual, expected))
        else:
            ok = (actual == expected)

    if ok:
        passed += 1
        print(f"  PASS  {test_name}")
    else:
        failed += 1
        msg = f"  FAIL  {test_name}: expected {expected!r}, got {actual!r}"
        print(msg)
        errors.append(msg)


def assert_2d_equal(test_name, actual, expected, tolerance=None):
    global passed, failed
    ok = False
    if len(actual) == len(expected):
        ok = True
        for row_a, row_e in zip(actual, expected):
            if len(row_a) != len(row_e):
                ok = False
                break
            if tolerance is not None:
                if not all(abs(a - e) <= tolerance for a, e in zip(row_a, row_e)):
                    ok = False
                    break
            else:
                if row_a != row_e:
                    ok = False
                    break

    if ok:
        passed += 1
        print(f"  PASS  {test_name}")
    else:
        failed += 1
        msg = f"  FAIL  {test_name}: expected {expected!r}, got {actual!r}"
        print(msg)
        errors.append(msg)


def assert_true(test_name, value):
    assert_equal(test_name, value, True)


def assert_false(test_name, value):
    assert_equal(test_name, value, False)


def assert_raises(test_name, fn):
    global passed, failed
    try:
        fn()
    except Exception:
        passed += 1
        print(f"  PASS  {test_name}")
        return
    failed += 1
    message = f"  FAIL  {test_name}: expected an exception, none raised"
    print(message)
    errors.append(message)


# ===========================================================================
#  Test suites -- helper methods
# ===========================================================================

def test_string_value_to_var_type():
    print("\n-- stringValueToVarType --")
    s = State()

    for name, expected in [
        ("DevBoolean", CmdArgType.DevBoolean),
        ("DevLong", CmdArgType.DevLong),
        ("DevDouble", CmdArgType.DevDouble),
        ("DevFloat", CmdArgType.DevFloat),
        ("DevString", CmdArgType.DevString),
    ]:
        got = Mqtt.stringValueToVarType(s, name)
        assert_equal(f"varType {name}", got, expected)

    # unsupported raises
    global passed, failed
    try:
        Mqtt.stringValueToVarType(s, "DevInvalid")
        failed += 1
        errors.append("  FAIL  varType invalid: expected exception")
        print("  FAIL  varType invalid: expected exception")
    except Exception:
        passed += 1
        print("  PASS  varType invalid raises")


def test_string_value_to_write_type():
    print("\n-- stringValueToWriteType --")
    s = State()

    for name, expected in [
        ("READ", AttrWriteType.READ),
        ("WRITE", AttrWriteType.WRITE),
        ("READ_WRITE", AttrWriteType.READ_WRITE),
    ]:
        got = Mqtt.stringValueToWriteType(s, name)
        assert_equal(f"writeType {name}", got, expected)

    # READ_WITH_WRITE needs an associated write attribute, which this driver never defines: building
    # the Attr anyway aborts init_device and takes the whole device server down, so it has to be
    # turned away here instead of being mapped through
    assert_raises("writeType READ_WITH_WRITE rejected",
                  lambda: Mqtt.stringValueToWriteType(s, "READ_WITH_WRITE"))
    assert_raises("writeType unknown rejected",
                  lambda: Mqtt.stringValueToWriteType(s, "NOPE"))


def test_string_value_to_format_type():
    print("\n-- stringValueToFormatType --")
    s = State()

    for name, expected in [
        ("SCALAR", AttrDataFormat.SCALAR),
        ("SPECTRUM", AttrDataFormat.SPECTRUM),
        ("IMAGE", AttrDataFormat.IMAGE),
    ]:
        got = Mqtt.stringValueToFormatType(s, name)
        assert_equal(f"formatType {name}", got, expected)

    # unknown defaults to SCALAR
    got = Mqtt.stringValueToFormatType(s, "")
    assert_equal("formatType empty -> SCALAR", got, AttrDataFormat.SCALAR)

    got = Mqtt.stringValueToFormatType(s, "BOGUS")
    assert_equal("formatType bogus -> SCALAR", got, AttrDataFormat.SCALAR)


def test_string_value_to_float():
    print("\n-- stringValueToFloat --")
    s = State()

    assert_equal("float '3.14'", Mqtt.stringValueToFloat(s, "3.14"), 3.14)
    assert_equal("float '0'", Mqtt.stringValueToFloat(s, "0"), 0.0)
    assert_equal("float '-1.5'", Mqtt.stringValueToFloat(s, "-1.5"), -1.5)
    assert_equal("float ''", Mqtt.stringValueToFloat(s, ""), 0.0)
    assert_equal("float None", Mqtt.stringValueToFloat(s, None), 0.0)


def test_cast_element():
    print("\n-- _cast_element --")
    s = State()

    assert_equal("cast bool True", Mqtt._cast_element(s, 1, CmdArgType.DevBoolean), True)
    assert_equal("cast bool False", Mqtt._cast_element(s, 0, CmdArgType.DevBoolean), False)
    assert_equal("cast long", Mqtt._cast_element(s, 3.7, CmdArgType.DevLong), 3)
    assert_equal("cast float", Mqtt._cast_element(s, 5, CmdArgType.DevFloat), 5.0)
    assert_equal("cast double", Mqtt._cast_element(s, 5, CmdArgType.DevDouble), 5.0)
    assert_equal("cast string passthrough", Mqtt._cast_element(s, "abc", CmdArgType.DevString), "abc")


def test_cast_array_1d():
    print("\n-- _cast_array 1D --")
    s = State()

    got = Mqtt._cast_array(s, [1, 2, 3], CmdArgType.DevFloat)
    assert_list_equal("1D float", got, [1.0, 2.0, 3.0])

    got = Mqtt._cast_array(s, [1.9, 2.1, 3.7], CmdArgType.DevLong)
    assert_list_equal("1D long", got, [1, 2, 3])

    got = Mqtt._cast_array(s, [1, 0, 1], CmdArgType.DevBoolean)
    assert_list_equal("1D bool", got, [True, False, True])

    got = Mqtt._cast_array(s, [], CmdArgType.DevFloat)
    assert_list_equal("1D empty", got, [])


def test_cast_array_2d():
    print("\n-- _cast_array 2D --")
    s = State()

    got = Mqtt._cast_array(s, [[1, 2], [3, 4]], CmdArgType.DevFloat)
    assert_2d_equal("2D float", got, [[1.0, 2.0], [3.0, 4.0]])

    got = Mqtt._cast_array(s, [[1.9, 2.1], [3.7, 4.2]], CmdArgType.DevLong)
    assert_2d_equal("2D long", got, [[1, 2], [3, 4]])

    got = Mqtt._cast_array(s, [[1, 0], [0, 1]], CmdArgType.DevBoolean)
    assert_2d_equal("2D bool", got, [[True, False], [False, True]])


# ===========================================================================
#  Test suites -- scalar conversion via stringValueToTypeValue
# ===========================================================================

def test_scalar_devstring():
    print("\n-- scalar: DevString --")
    s = State()
    register_attr(s, "s_str", CmdArgType.DevString)

    assert_equal("string 'hello'", convert(s, "s_str", "hello"), "hello")
    assert_equal("string empty", convert(s, "s_str", ""), "")
    assert_equal("string bytes", convert(s, "s_str", b"from_mqtt"), "from_mqtt")


def test_scalar_devboolean():
    print("\n-- scalar: DevBoolean --")
    s = State()
    register_attr(s, "s_bool", CmdArgType.DevBoolean)

    assert_true("bool 'true'", convert(s, "s_bool", "true"))
    assert_true("bool 'True'", convert(s, "s_bool", "True"))
    assert_true("bool 'TRUE'", convert(s, "s_bool", "TRUE"))
    assert_false("bool 'false'", convert(s, "s_bool", "false"))
    assert_false("bool 'False'", convert(s, "s_bool", "False"))
    assert_true("bool '1'", convert(s, "s_bool", "1"))
    assert_false("bool '0'", convert(s, "s_bool", "0"))
    assert_true("bool bytes b'true'", convert(s, "s_bool", b"true"))
    assert_false("bool bytes b'false'", convert(s, "s_bool", b"false"))


def test_scalar_devlong():
    print("\n-- scalar: DevLong --")
    s = State()
    register_attr(s, "s_long", CmdArgType.DevLong)

    assert_equal("long '42'", convert(s, "s_long", "42"), 42)
    assert_equal("long '-1'", convert(s, "s_long", "-1"), -1)
    assert_equal("long '0'", convert(s, "s_long", "0"), 0)
    assert_equal("long '3.9' truncates", convert(s, "s_long", "3.9"), 3)
    assert_equal("long bytes b'99'", convert(s, "s_long", b"99"), 99)
    assert_equal("long empty -> 0", convert(s, "s_long", ""), 0)


def test_scalar_devfloat():
    print("\n-- scalar: DevFloat --")
    s = State()
    register_attr(s, "s_float", CmdArgType.DevFloat)

    assert_equal("float '3.14'", convert(s, "s_float", "3.14"), 3.14, tolerance=1e-5)
    assert_equal("float '-0.5'", convert(s, "s_float", "-0.5"), -0.5)
    assert_equal("float '0'", convert(s, "s_float", "0"), 0.0)
    assert_equal("float empty -> 0", convert(s, "s_float", ""), 0.0)
    assert_equal("float bytes", convert(s, "s_float", b"1.5"), 1.5)


def test_scalar_devdouble():
    print("\n-- scalar: DevDouble --")
    s = State()
    register_attr(s, "s_dbl", CmdArgType.DevDouble)

    assert_equal("double '2.718'", convert(s, "s_dbl", "2.718281828"), 2.718281828, tolerance=1e-9)
    assert_equal("double '-1e10'", convert(s, "s_dbl", "-1e10"), -1e10)
    assert_equal("double empty -> 0", convert(s, "s_dbl", ""), 0.0)


# ===========================================================================
#  Test suites -- spectrum (1D) conversion via stringValueToTypeValue
# ===========================================================================

def test_spectrum_devfloat():
    print("\n-- spectrum: DevFloat --")
    s = State()
    register_attr(s, "sp_float", CmdArgType.DevFloat, AttrDataFormat.SPECTRUM)

    payload = json.dumps([1.1, 2.2, 3.3])
    got = convert(s, "sp_float", payload)
    assert_list_equal("spectrum float", got, [1.1, 2.2, 3.3], tolerance=1e-6)

    # from bytes (MQTT payload)
    got = convert(s, "sp_float", payload.encode())
    assert_list_equal("spectrum float bytes", got, [1.1, 2.2, 3.3], tolerance=1e-6)

    # empty
    got = convert(s, "sp_float", "")
    assert_list_equal("spectrum float empty", got, [])


def test_spectrum_devdouble():
    print("\n-- spectrum: DevDouble --")
    s = State()
    register_attr(s, "sp_dbl", CmdArgType.DevDouble, AttrDataFormat.SPECTRUM)

    payload = json.dumps([1e-10, 2.718281828, -3.14159])
    got = convert(s, "sp_dbl", payload)
    assert_list_equal("spectrum double", got, [1e-10, 2.718281828, -3.14159], tolerance=1e-12)


def test_spectrum_devlong():
    print("\n-- spectrum: DevLong --")
    s = State()
    register_attr(s, "sp_long", CmdArgType.DevLong, AttrDataFormat.SPECTRUM)

    payload = json.dumps([10, 20, 30, -40])
    got = convert(s, "sp_long", payload)
    assert_list_equal("spectrum long", got, [10, 20, 30, -40])

    # JSON floats cast to int
    payload = json.dumps([1.9, 2.1, 3.7])
    got = convert(s, "sp_long", payload)
    assert_list_equal("spectrum long from floats", got, [1, 2, 3])


def test_spectrum_devboolean():
    print("\n-- spectrum: DevBoolean --")
    s = State()
    register_attr(s, "sp_bool", CmdArgType.DevBoolean, AttrDataFormat.SPECTRUM)

    payload = json.dumps([1, 0, 1, 0])
    got = convert(s, "sp_bool", payload)
    assert_list_equal("spectrum bool int", got, [True, False, True, False])

    # JSON true/false
    payload = '[true, false, true]'
    got = convert(s, "sp_bool", payload)
    assert_list_equal("spectrum bool json", got, [True, False, True])


def test_spectrum_devstring():
    print("\n-- spectrum: DevString --")
    s = State()
    register_attr(s, "sp_str", CmdArgType.DevString, AttrDataFormat.SPECTRUM)

    payload = json.dumps(["hello", "world", "test"])
    got = convert(s, "sp_str", payload)
    assert_list_equal("spectrum string", got, ["hello", "world", "test"])

    # empty array
    payload = json.dumps([])
    got = convert(s, "sp_str", payload)
    assert_list_equal("spectrum string empty", got, [])


# ===========================================================================
#  Test suites -- image (2D) conversion via stringValueToTypeValue
# ===========================================================================

def test_image_devfloat():
    print("\n-- image: DevFloat --")
    s = State()
    register_attr(s, "img_float", CmdArgType.DevFloat, AttrDataFormat.IMAGE)

    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    payload = json.dumps(data)
    got = convert(s, "img_float", payload)
    assert_2d_equal("image float", got, data, tolerance=1e-6)

    # from bytes
    got = convert(s, "img_float", payload.encode())
    assert_2d_equal("image float bytes", got, data, tolerance=1e-6)


def test_image_devdouble():
    print("\n-- image: DevDouble --")
    s = State()
    register_attr(s, "img_dbl", CmdArgType.DevDouble, AttrDataFormat.IMAGE)

    data = [[1.111, 2.222], [3.333, 4.444], [5.555, 6.666]]
    payload = json.dumps(data)
    got = convert(s, "img_dbl", payload)
    assert_2d_equal("image double", got, data, tolerance=1e-9)


def test_image_devlong():
    print("\n-- image: DevLong --")
    s = State()
    register_attr(s, "img_long", CmdArgType.DevLong, AttrDataFormat.IMAGE)

    data = [[1, 2], [3, 4]]
    payload = json.dumps(data)
    got = convert(s, "img_long", payload)
    assert_2d_equal("image long", got, data)

    # floats cast to int
    payload = json.dumps([[1.9, 2.1], [3.7, 4.2]])
    got = convert(s, "img_long", payload)
    assert_2d_equal("image long from floats", got, [[1, 2], [3, 4]])


def test_image_devboolean():
    print("\n-- image: DevBoolean --")
    s = State()
    register_attr(s, "img_bool", CmdArgType.DevBoolean, AttrDataFormat.IMAGE)

    payload = '[[true, false], [false, true]]'
    got = convert(s, "img_bool", payload)
    assert_2d_equal("image bool", got, [[True, False], [False, True]])


# ===========================================================================
#  Test suites -- write serialization round-trips
# ===========================================================================

def test_write_roundtrip_scalar():
    print("\n-- write round-trip: scalar --")
    s = State()

    register_attr(s, "wr_float", CmdArgType.DevFloat)
    serialized = serialize_write(s, "wr_float", 3.14)
    got = convert(s, "wr_float", serialized)
    assert_equal("scalar float round-trip", got, 3.14, tolerance=1e-5)

    register_attr(s, "wr_long", CmdArgType.DevLong)
    serialized = serialize_write(s, "wr_long", 42)
    got = convert(s, "wr_long", serialized)
    assert_equal("scalar long round-trip", got, 42)

    register_attr(s, "wr_str", CmdArgType.DevString)
    serialized = serialize_write(s, "wr_str", "hello")
    got = convert(s, "wr_str", serialized)
    assert_equal("scalar string round-trip", got, "hello")

    register_attr(s, "wr_bool", CmdArgType.DevBoolean)
    serialized = serialize_write(s, "wr_bool", True)
    got = convert(s, "wr_bool", serialized)
    assert_true("scalar bool True round-trip", got)

    serialized = serialize_write(s, "wr_bool", False)
    got = convert(s, "wr_bool", serialized)
    assert_false("scalar bool False round-trip", got)


def test_write_roundtrip_spectrum():
    print("\n-- write round-trip: spectrum (numpy -> json -> parse) --")
    s = State()

    register_attr(s, "wr_sp_float", CmdArgType.DevFloat, AttrDataFormat.SPECTRUM)
    arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    serialized = serialize_write(s, "wr_sp_float", arr)
    got = convert(s, "wr_sp_float", serialized)
    assert_list_equal("spectrum float round-trip", got, [1.5, 2.5, 3.5], tolerance=1e-5)

    register_attr(s, "wr_sp_dbl", CmdArgType.DevDouble, AttrDataFormat.SPECTRUM)
    arr = np.array([1e-10, 2.718, -3.14], dtype=np.float64)
    serialized = serialize_write(s, "wr_sp_dbl", arr)
    got = convert(s, "wr_sp_dbl", serialized)
    assert_list_equal("spectrum double round-trip", got, [1e-10, 2.718, -3.14], tolerance=1e-9)

    register_attr(s, "wr_sp_long", CmdArgType.DevLong, AttrDataFormat.SPECTRUM)
    arr = np.array([10, -20, 30], dtype=np.int32)
    serialized = serialize_write(s, "wr_sp_long", arr)
    got = convert(s, "wr_sp_long", serialized)
    assert_list_equal("spectrum long round-trip", got, [10, -20, 30])

    register_attr(s, "wr_sp_bool", CmdArgType.DevBoolean, AttrDataFormat.SPECTRUM)
    arr = np.array([True, False, True])
    serialized = serialize_write(s, "wr_sp_bool", arr)
    got = convert(s, "wr_sp_bool", serialized)
    assert_list_equal("spectrum bool round-trip", got, [True, False, True])

    # a DevString spectrum is handed over as a plain python list rather than a numpy array, which
    # used to hit "'list' object has no attribute 'tolist'" and fail every write of one
    register_attr(s, "wr_sp_str", CmdArgType.DevString, AttrDataFormat.SPECTRUM)
    serialized = serialize_write(s, "wr_sp_str", ["cherry", "banana", ""])
    got = convert(s, "wr_sp_str", serialized)
    assert_list_equal("spectrum string round-trip", got, ["cherry", "banana", ""])


def test_write_roundtrip_image_string():
    print("\n-- write round-trip: image of strings (list of lists, no numpy) --")
    s = State()

    register_attr(s, "wr_img_str", CmdArgType.DevString, AttrDataFormat.IMAGE)
    rows = [["a", "b", "c"], ["d", "e", "f"]]
    serialized = serialize_write(s, "wr_img_str", rows)
    assert_equal("image string serialises", serialized, json.dumps(rows))

    # the numeric image stays on the numpy path, so both have to keep working
    register_attr(s, "wr_img_dbl", CmdArgType.DevDouble, AttrDataFormat.IMAGE)
    arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    serialized = serialize_write(s, "wr_img_dbl", arr)
    assert_equal("image double serialises", serialized, json.dumps([[1.5, 2.5], [3.5, 4.5]]))


def test_write_roundtrip_image():
    print("\n-- write round-trip: image (numpy 2D -> json -> parse) --")
    s = State()

    register_attr(s, "wr_img_float", CmdArgType.DevFloat, AttrDataFormat.IMAGE)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    serialized = serialize_write(s, "wr_img_float", arr)
    got = convert(s, "wr_img_float", serialized)
    assert_2d_equal("image float round-trip", got, [[1.0, 2.0], [3.0, 4.0]], tolerance=1e-5)

    register_attr(s, "wr_img_dbl", CmdArgType.DevDouble, AttrDataFormat.IMAGE)
    arr = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    serialized = serialize_write(s, "wr_img_dbl", arr)
    got = convert(s, "wr_img_dbl", serialized)
    assert_2d_equal("image double round-trip", got, [[1.1, 2.2], [3.3, 4.4]], tolerance=1e-9)

    register_attr(s, "wr_img_long", CmdArgType.DevLong, AttrDataFormat.IMAGE)
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    serialized = serialize_write(s, "wr_img_long", arr)
    got = convert(s, "wr_img_long", serialized)
    assert_2d_equal("image long round-trip", got, [[1, 2, 3], [4, 5, 6]])

    register_attr(s, "wr_img_bool", CmdArgType.DevBoolean, AttrDataFormat.IMAGE)
    arr = np.array([[True, False], [False, True]])
    serialized = serialize_write(s, "wr_img_bool", arr)
    got = convert(s, "wr_img_bool", serialized)
    assert_2d_equal("image bool round-trip", got, [[True, False], [False, True]])


# ===========================================================================
#  Test suites -- edge cases
# ===========================================================================

def test_edge_cases():
    print("\n-- edge cases --")
    s = State()

    # Large spectrum
    register_attr(s, "edge_big", CmdArgType.DevFloat, AttrDataFormat.SPECTRUM)
    big = list(range(256))
    payload = json.dumps(big)
    got = convert(s, "edge_big", payload)
    assert_equal("large spectrum length", len(got), 256)
    assert_equal("large spectrum first", got[0], 0.0)
    assert_equal("large spectrum last", got[255], 255.0)

    # Large image
    register_attr(s, "edge_big_img", CmdArgType.DevDouble, AttrDataFormat.IMAGE)
    img = [[float(r * 10 + c) for c in range(10)] for r in range(10)]
    payload = json.dumps(img)
    got = convert(s, "edge_big_img", payload)
    assert_equal("large image rows", len(got), 10)
    assert_equal("large image cols", len(got[0]), 10)
    assert_equal("large image [9][9]", got[9][9], 99.0)

    # Spectrum with single element
    register_attr(s, "edge_one", CmdArgType.DevLong, AttrDataFormat.SPECTRUM)
    got = convert(s, "edge_one", "[42]")
    assert_list_equal("single-element spectrum", got, [42])

    # None for non-scalar returns empty
    register_attr(s, "edge_none", CmdArgType.DevFloat, AttrDataFormat.SPECTRUM)
    got = convert(s, "edge_none", None)
    assert_list_equal("spectrum None", got, [])

    got = convert(s, "edge_none", "")
    assert_list_equal("spectrum empty string", got, [])

    # Scalar boolean edge: '0' and '1'
    register_attr(s, "edge_bool", CmdArgType.DevBoolean)
    assert_false("bool '0'", convert(s, "edge_bool", "0"))
    assert_true("bool '1'", convert(s, "edge_bool", "1"))

    # Scalar long from empty string
    register_attr(s, "edge_long", CmdArgType.DevLong)
    assert_equal("long empty -> 0", convert(s, "edge_long", ""), 0)

    # Numpy write of large spectrum
    register_attr(s, "edge_np", CmdArgType.DevDouble, AttrDataFormat.SPECTRUM)
    arr = np.arange(100, dtype=np.float64)
    serialized = serialize_write(s, "edge_np", arr)
    got = convert(s, "edge_np", serialized)
    assert_equal("numpy large spectrum len", len(got), 100)
    assert_equal("numpy large spectrum [99]", got[99], 99.0)

    # Numpy write of image with negative values
    register_attr(s, "edge_np_img", CmdArgType.DevFloat, AttrDataFormat.IMAGE)
    arr = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    serialized = serialize_write(s, "edge_np_img", arr)
    got = convert(s, "edge_np_img", serialized)
    assert_2d_equal("numpy negative image", got, [[-1.0, -2.0], [-3.0, -4.0]], tolerance=1e-5)


# ===========================================================================
#  Test suites -- qos / retain / protocol descriptor parsing
# ===========================================================================

def test_qos_parsing():
    print("\n-- stringValueToQos --")
    s = State()
    s.default_qos = 1

    assert_equal("qos omitted -> default_qos", Mqtt.stringValueToQos(s, ""), 1)
    assert_equal("qos None -> default_qos", Mqtt.stringValueToQos(s, None), 1)
    assert_equal("qos 0 explicit beats default", Mqtt.stringValueToQos(s, 0), 0)
    assert_equal("qos '2' from a string descriptor", Mqtt.stringValueToQos(s, "2"), 2)
    assert_raises("qos 3 rejected", lambda: Mqtt.stringValueToQos(s, 3))
    assert_raises("qos -1 rejected", lambda: Mqtt.stringValueToQos(s, -1))
    assert_raises("qos non numeric rejected", lambda: Mqtt.stringValueToQos(s, "high"))


def test_bool_parsing():
    print("\n-- stringValueToBool --")
    s = State()

    assert_true("retain json true", Mqtt.stringValueToBool(s, True, False))
    assert_false("retain json false", Mqtt.stringValueToBool(s, False, True))
    assert_true("retain 'true'", Mqtt.stringValueToBool(s, "true", False))
    assert_true("retain 'True'", Mqtt.stringValueToBool(s, "True", False))
    assert_true("retain '1'", Mqtt.stringValueToBool(s, "1", False))
    assert_false("retain '0'", Mqtt.stringValueToBool(s, "0", True))
    assert_false("retain 'no'", Mqtt.stringValueToBool(s, "no", True))
    assert_true("retain omitted -> default True", Mqtt.stringValueToBool(s, "", True))
    assert_false("retain omitted -> default False", Mqtt.stringValueToBool(s, "", False))
    assert_true("retain None -> default", Mqtt.stringValueToBool(s, None, True))


def test_protocol_mapping():
    print("\n-- stringValueToProtocol --")
    s = State()

    assert_equal("protocol 3.1", Mqtt.stringValueToProtocol(s, "3.1"), mqtt.MQTTv31)
    assert_equal("protocol 3.1.1", Mqtt.stringValueToProtocol(s, "3.1.1"), mqtt.MQTTv311)
    assert_equal("protocol 5", Mqtt.stringValueToProtocol(s, "5"), mqtt.MQTTv5)
    assert_equal("protocol 5.0", Mqtt.stringValueToProtocol(s, "5.0"), mqtt.MQTTv5)
    # the README promises exactly these three versions, so an unknown one is named rather than
    # silently downgraded to the paho default
    assert_raises("protocol 4 rejected", lambda: Mqtt.stringValueToProtocol(s, "4"))
    assert_raises("protocol empty rejected", lambda: Mqtt.stringValueToProtocol(s, ""))


def test_connect_result_text():
    print("\n-- connect_result_text --")
    s = State()

    assert_equal("rc 0 text", Mqtt.connect_result_text(s, 0), mqtt.connack_string(0))
    assert_equal("rc 5 text", Mqtt.connect_result_text(s, 5), mqtt.connack_string(5))
    # mqtt 5 delivers a ReasonCodes object instead of an int and renders its own text
    reason = mqtt.ReasonCodes(mqtt.CONNACK >> 4, identifier=135)
    assert_equal("mqtt5 reason code text", Mqtt.connect_result_text(s, reason), str(reason))


# ===========================================================================
#  Test suites -- json payload field extraction
# ===========================================================================

def test_apply_modifier():
    print("\n-- apply_modifier --")
    s = State()

    # no modifier: the payload passes through, bytes decoded to str
    assert_equal("no modifier str", Mqtt.apply_modifier(s, "raw", ""), "raw")
    assert_equal("no modifier bytes", Mqtt.apply_modifier(s, b"raw", ""), "raw")

    payload = b'{"temp": 21.5, "hum": 40, "ok": true, "name": "probe", "missing": null}'
    assert_equal("field float", Mqtt.apply_modifier(s, payload, "temp"), "21.5")
    assert_equal("field int", Mqtt.apply_modifier(s, payload, "hum"), "40")
    assert_equal("field bool keeps json spelling", Mqtt.apply_modifier(s, payload, "ok"), "true")
    assert_equal("field string", Mqtt.apply_modifier(s, payload, "name"), "probe")
    assert_equal("field null -> empty", Mqtt.apply_modifier(s, payload, "missing"), "")

    nested = b'{"sensor": {"temp": 3.5}, "values": [10, 20, 30]}'
    assert_equal("nested path", Mqtt.apply_modifier(s, nested, "sensor.temp"), "3.5")
    assert_equal("list index", Mqtt.apply_modifier(s, nested, "values.1"), "20")
    # an array field stays json so the spectrum conversion downstream can parse it again
    assert_equal("array field stays json", Mqtt.apply_modifier(s, nested, "values"),
                 json.dumps([10, 20, 30]))

    assert_raises("missing key raises", lambda: Mqtt.apply_modifier(s, payload, "nope"))
    assert_raises("non json payload raises", lambda: Mqtt.apply_modifier(s, b"plain", "temp"))

    # a binary payload carries no json field and must not blow up on decode
    assert_equal("binary payload passes through", Mqtt.apply_modifier(s, b"\xff\xfe", ""), b"\xff\xfe")


def test_modifier_spectrum():
    print("\n-- modifier feeding a spectrum --")
    s = State()
    register_attr(s, "series", CmdArgType.DevDouble, AttrDataFormat.SPECTRUM,
                  topic="sensors/a", modifier="values")

    extracted = Mqtt.apply_modifier(s, b'{"values": [1.5, 2.5], "unit": "C"}', "values")
    got = convert(s, "series", extracted)
    assert_list_equal("spectrum out of a json field", got, [1.5, 2.5], tolerance=1e-9)


# ===========================================================================
#  Test suites -- message dispatch
# ===========================================================================

def test_on_message_dispatch():
    print("\n-- on_message: one topic, several attributes --")
    s = State()
    for name, modifier in (("temp", "temp"), ("hum", "hum")):
        Mqtt.add_dynamic_attribute(s, name, variable_type_name="DevDouble",
                                   write_type_name="READ", topic="sensors/a", modifier=modifier)
        s._device_attr.register(name, CmdArgType.DevDouble, AttrDataFormat.SCALAR)

    assert_list_equal("topic index holds both attributes", s.topicAttributes["sensors/a"],
                      ["temp", "hum"])

    Mqtt.on_message(s, None, None, MockMessage("sensors/a", b'{"temp": 21.5, "hum": 40}'))
    assert_list_equal("both attributes pushed", s.events, [("temp", 21.5), ("hum", 40.0)])

    # an identical payload changes nothing, so nothing is pushed again
    s.events = []
    Mqtt.on_message(s, None, None, MockMessage("sensors/a", b'{"temp": 21.5, "hum": 40}'))
    assert_equal("unchanged payload pushes nothing", len(s.events), 0)

    # only the field that actually moved is pushed
    Mqtt.on_message(s, None, None, MockMessage("sensors/a", b'{"temp": 22.0, "hum": 40}'))
    assert_list_equal("only the changed field pushed", s.events, [("temp", 22.0)])

    assert_equal("last_msg_at recorded", s._last_msg_at != "none", True)


def test_on_message_bad_modifier_is_isolated():
    print("\n-- on_message: a modifier that misses keeps the others alive --")
    s = State()
    for name, modifier in (("temp", "temp"), ("hum", "hum")):
        Mqtt.add_dynamic_attribute(s, name, variable_type_name="DevDouble",
                                   write_type_name="READ", topic="sensors/a", modifier=modifier)
        s._device_attr.register(name, CmdArgType.DevDouble, AttrDataFormat.SCALAR)

    # hum is absent from this payload; temp still has to get through
    Mqtt.on_message(s, None, None, MockMessage("sensors/a", b'{"temp": 7.0}'))
    assert_list_equal("surviving attribute still pushed", s.events, [("temp", 7.0)])
    assert_equal("failing attribute keeps its old value", s.dynamicAttributes["hum"]["value"], "")


def test_on_message_wildcard_topic():
    print("\n-- on_message: topic seen through a wildcard subscription --")
    s = State()
    Mqtt.add_dynamic_attribute(s, "sensors/#")
    s._connected = True
    s.client.subscriptions = []

    Mqtt.on_message(s, None, None, MockMessage("sensors/a", b"hello"))
    assert_true("attribute created for the concrete topic", "sensors/a" in s.dynamicAttributes)
    assert_list_equal("value pushed", s.events, [("sensors/a", "hello")])
    # subscribing again would overlap the wildcard, which lets the broker deliver every message twice
    assert_equal("no overlapping subscription", len(s.client.subscriptions), 0)


# ===========================================================================
#  Test suites -- subscribe / publish
# ===========================================================================

def test_subscribe_qos():
    print("\n-- subscribe --")
    s = State()
    s.default_qos = 0
    register_attr(s, "a", CmdArgType.DevDouble, topic="sensors/a", qos=1)
    register_attr(s, "b", CmdArgType.DevDouble, topic="sensors/a", qos=2)
    register_attr(s, "c", CmdArgType.DevDouble, topic="sensors/c", qos=0)

    Mqtt.subscribe(s, "sensors/a")
    # one subscription serves both attributes, so it has to carry the strongest qos either asked for
    assert_equal("strongest qos wins", s.client.subscriptions[-1], ("sensors/a", 2))

    Mqtt.subscribe(s, "sensors/c")
    assert_equal("qos 0 stays 0", s.client.subscriptions[-1], ("sensors/c", 0))

    # a topic with no attribute behind it falls back to the device default
    s.default_qos = 1
    Mqtt.subscribe(s, "sensors/unknown")
    assert_equal("unknown topic uses default_qos", s.client.subscriptions[-1], ("sensors/unknown", 1))


def test_publish_command():
    print("\n-- publish command --")
    s = State()

    Mqtt.publish(s, ["t/a", "5"])
    assert_equal("two argument form", s.published[-1], ("t/a", "5", 0, False))

    Mqtt.publish(s, ["t/a", "5", "2"])
    assert_equal("qos argument", s.published[-1], ("t/a", "5", 2, False))

    Mqtt.publish(s, ["t/a", "5", "1", "true"])
    assert_equal("qos and retain arguments", s.published[-1], ("t/a", "5", 1, True))

    s.default_qos = 2
    s.default_retain = True
    Mqtt.publish(s, ["t/a", "5"])
    assert_equal("device defaults apply", s.published[-1], ("t/a", "5", 2, True))

    # and the value really reaches paho with that qos and retain flag - State stubs publish_value
    # for every other test here, so this one calls the real one
    Mqtt.publish_value(s, "t/b", "7", 1, True)
    assert_equal("handed to the client", s.client.published[-1], ("t/b", "7", 1, True))


def test_write_uses_topic_qos_retain():
    print("\n-- write: publishes on the topic with its qos and retain --")
    s = State()
    register_attr(s, "setpoint", CmdArgType.DevDouble, topic="plant/setpoint", qos=2, retain=True)

    serialize_write(s, "setpoint", 21.5)
    assert_equal("published to the topic, not the attribute name", s.published[-1],
                 ("plant/setpoint", "21.5", 2, True))
    assert_list_equal("write pushes a change event", s.events, [("setpoint", 21.5)])


def test_modifier_write_rejected():
    print("\n-- write: a json field is not writable --")
    s = State()
    register_attr(s, "temp", CmdArgType.DevDouble, topic="sensors/a", modifier="temp")

    # publishing back would have to rebuild the whole document, and the other fields are not ours
    assert_raises("write to a modifier attribute rejected",
                  lambda: serialize_write(s, "temp", 5.0))
    assert_equal("nothing published", len(s.published), 0)


# ===========================================================================
#  Test suites -- attribute creation and connection handling
# ===========================================================================

def test_add_dynamic_attribute_binding():
    print("\n-- add_dynamic_attribute --")
    s = State()

    Mqtt.add_dynamic_attribute(s, "temperature", variable_type_name="DevDouble",
                               write_type_name="READ", topic="sensors/a", modifier="temp",
                               qos="1", retain="true")
    config = s.dynamicAttributes["temperature"]
    assert_equal("topic kept apart from the name", config["topic"], "sensors/a")
    assert_equal("modifier stored", config["modifier"], "temp")
    assert_equal("qos parsed", config["qos"], 1)
    assert_true("retain parsed", config["retain"])
    assert_list_equal("topic index updated", s.topicAttributes["sensors/a"], ["temperature"])

    # the topic defaults to the attribute name, which is what every descriptor without one relies on
    Mqtt.add_dynamic_attribute(s, "plain/topic")
    assert_equal("name doubles as topic", s.dynamicAttributes["plain/topic"]["topic"], "plain/topic")

    # a repeated name is ignored rather than registered twice with tango
    Mqtt.add_dynamic_attribute(s, "plain/topic")
    assert_list_equal("no duplicate in the topic index", s.topicAttributes["plain/topic"],
                      ["plain/topic"])

    # an empty name is a no-op, the comma separated fallback list can produce one
    before = len(s.dynamicAttributes)
    Mqtt.add_dynamic_attribute(s, "")
    assert_equal("empty name ignored", len(s.dynamicAttributes), before)


def test_add_dynamic_attribute_subscribes_when_connected():
    print("\n-- add_dynamic_attribute: subscribes at runtime --")
    s = State()

    # before the connection is up on_connect will subscribe everything, so this one must not
    Mqtt.add_dynamic_attribute(s, "early")
    assert_equal("no subscription while disconnected", len(s.client.subscriptions), 0)

    s._connected = True
    Mqtt.add_dynamic_attribute(s, "late", qos="1")
    assert_equal("subscribed straight away", s.client.subscriptions[-1], ("late", 1))


def test_on_connect():
    print("\n-- on_connect --")
    s = State()
    register_attr(s, "a", CmdArgType.DevDouble, topic="sensors/a", qos=1)
    register_attr(s, "b", CmdArgType.DevDouble, topic="sensors/b", qos=0)

    Mqtt.on_connect(s, None, None, {}, 0)
    assert_equal("state ON", s.state, DevState.ON)
    assert_true("connected flag set", s._connected)
    assert_list_equal("every topic subscribed", sorted(s.client.subscriptions),
                      [("sensors/a", 1), ("sensors/b", 0)])

    # a refused connack used to be reported as ON, which hid a wrong password behind a healthy device
    s = State()
    register_attr(s, "a", CmdArgType.DevDouble, topic="sensors/a")
    Mqtt.on_connect(s, None, None, {}, 5)
    assert_equal("state FAULT on refusal", s.state, DevState.FAULT)
    assert_false("connected flag cleared", s._connected)
    assert_equal("nothing subscribed", len(s.client.subscriptions), 0)
    assert_true("status names the reason", mqtt.connack_string(5) in s.status)


def test_birth_message():
    print("\n-- on_connect: birth message --")
    s = State()
    s.will_topic = "plant/status"
    s.will_payload = "offline"
    s.will_retain = True
    s.will_qos = 1

    Mqtt.on_connect(s, None, None, {}, 0)
    # without it the retained will payload stays the last word on that topic forever
    assert_equal("birth published on the will topic", s.published[-1],
                 ("plant/status", "online", 1, True))

    # an explicit birth topic wins over the will topic
    s = State()
    s.will_topic = "plant/status"
    s.birth_topic = "plant/online"
    Mqtt.on_connect(s, None, None, {}, 0)
    assert_equal("explicit birth topic used", s.published[-1][0], "plant/online")

    # no will and no birth topic means no birth message at all
    s = State()
    Mqtt.on_connect(s, None, None, {}, 0)
    assert_equal("nothing published without a will", len(s.published), 0)


def test_on_disconnect():
    print("\n-- on_disconnect --")
    s = State()
    s._connected = True
    s._refused = False

    Mqtt.on_disconnect(s, None, None, 1)
    assert_false("connected flag cleared", s._connected)
    assert_equal("state UNKNOWN", s.state, DevState.UNKNOWN)

    # a disconnect we asked for is not a fault, and paho must not be told to reconnect
    s = State()
    s._connected = True
    s._refused = False
    Mqtt.on_disconnect(s, None, None, 0)
    assert_false("connected flag cleared on clean disconnect", s._connected)
    assert_equal("clean disconnect leaves the state alone", s.state, None)

    # paho follows a refused connack with a disconnect callback carrying the same code; on_connect
    # has already recorded why, and this must not overwrite it with a generic reconnect notice
    s = State()
    Mqtt.on_connect(s, None, None, {}, 5)
    refusal_status = s.status
    Mqtt.on_disconnect(s, None, None, 5)
    assert_equal("refusal keeps the FAULT state", s.state, DevState.FAULT)
    assert_equal("refusal keeps its status text", s.status, refusal_status)


# ===========================================================================
#  Main
# ===========================================================================

def main():
    global passed, failed

    # -- helper method tests --
    test_string_value_to_var_type()
    test_string_value_to_write_type()
    test_string_value_to_format_type()
    test_string_value_to_float()
    test_cast_element()
    test_cast_array_1d()
    test_cast_array_2d()

    # -- scalar type conversion --
    test_scalar_devstring()
    test_scalar_devboolean()
    test_scalar_devlong()
    test_scalar_devfloat()
    test_scalar_devdouble()

    # -- spectrum (1D) type conversion --
    test_spectrum_devfloat()
    test_spectrum_devdouble()
    test_spectrum_devlong()
    test_spectrum_devboolean()
    test_spectrum_devstring()

    # -- image (2D) type conversion --
    test_image_devfloat()
    test_image_devdouble()
    test_image_devlong()
    test_image_devboolean()

    # -- write serialization round-trips --
    test_write_roundtrip_scalar()
    test_write_roundtrip_spectrum()
    test_write_roundtrip_image_string()
    test_write_roundtrip_image()

    # -- edge cases --
    test_edge_cases()

    # -- qos / retain / protocol descriptor parsing --
    test_qos_parsing()
    test_bool_parsing()
    test_protocol_mapping()
    test_connect_result_text()

    # -- json payload field extraction --
    test_apply_modifier()
    test_modifier_spectrum()

    # -- message dispatch --
    test_on_message_dispatch()
    test_on_message_bad_modifier_is_isolated()
    test_on_message_wildcard_topic()

    # -- subscribe / publish --
    test_subscribe_qos()
    test_publish_command()
    test_write_uses_topic_qos_retain()
    test_modifier_write_rejected()

    # -- attribute creation and connection handling --
    test_add_dynamic_attribute_binding()
    test_add_dynamic_attribute_subscribes_when_connected()
    test_on_connect()
    test_birth_message()
    test_on_disconnect()

    # -- summary --
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\n  Failures:")
        for e in errors:
            print(f"    {e}")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
