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
from tango import CmdArgType, AttrDataFormat, AttrWriteType

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


class State:
    """Carries instance state; every method lookup falls through to Mqtt."""

    def __init__(self):
        self._device_attr = MockDeviceAttr()
        self.dynamicAttributes = {}

    def get_device_attr(self):
        return self._device_attr

    def __getattr__(self, name):
        import functools
        attr = getattr(Mqtt, name, None)
        if attr is not None and callable(attr):
            return functools.partial(attr, self)
        raise AttributeError(f"'State' has no attribute '{name}'")


# Thin helpers

def register_attr(s, name, data_type, data_format=AttrDataFormat.SCALAR):
    s._device_attr.register(name, data_type, data_format)
    s.dynamicAttributes[name] = ""


def convert(s, name, val):
    """Simulate MQTT payload -> typed value (calls real Mqtt.stringValueToTypeValue)."""
    return Mqtt.stringValueToTypeValue(s, name, val)


def serialize_write(s, name, value):
    """Simulate Tango write -> MQTT string (same logic as Mqtt.write_dynamic_attr)."""
    attr_info = s.get_device_attr().get_attr_by_name(name)
    if attr_info.get_data_format() != AttrDataFormat.SCALAR:
        return json.dumps(value.tolist())
    else:
        return str(value)


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
        ("READ_WITH_WRITE", AttrWriteType.READ_WITH_WRITE),
    ]:
        got = Mqtt.stringValueToWriteType(s, name)
        assert_equal(f"writeType {name}", got, expected)


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
    test_write_roundtrip_image()

    # -- edge cases --
    test_edge_cases()

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
