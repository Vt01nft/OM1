import asyncio
import importlib
import sys
import types

import pytest


@pytest.fixture
def serial_connector_module(monkeypatch):
    """Import serial connector module with optional deps stubbed."""

    pydantic_module = types.ModuleType("pydantic")

    class StubBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    pydantic_module.BaseModel = StubBaseModel
    pydantic_module.ConfigDict = lambda **kwargs: kwargs
    pydantic_module.Field = lambda default=None, description="": default
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_module)

    serial_module = types.ModuleType("serial")

    class StubSerial:
        def __init__(self, port, baudrate):
            self.port = port
            self.baudrate = baudrate
            self.is_open = True
            self.writes = []

        def write(self, data):
            self.writes.append(data)

    serial_module.Serial = StubSerial
    monkeypatch.setitem(sys.modules, "serial", serial_module)

    module_name = "actions.move_serial_arduino.connector.serial_arduino"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


class FakeSerial:
    def __init__(self, is_open=True):
        self.is_open = is_open
        self.writes = []

    def write(self, data):
        self.writes.append(data)


def test_init_opens_serial_when_port_is_set(serial_connector_module):
    config = serial_connector_module.MoveSerialConfig(port="COM3")

    connector = serial_connector_module.MoveSerialConnector(config)

    assert connector.ser is not None
    assert connector.ser.port == "COM3"
    assert connector.ser.baudrate == 9600


def test_connect_writes_expected_command_for_known_action(serial_connector_module):
    config = serial_connector_module.MoveSerialConfig(port="")
    connector = serial_connector_module.MoveSerialConnector(config)
    fake_serial = FakeSerial(is_open=True)
    connector.ser = fake_serial

    output_interface = types.SimpleNamespace(action="medium jump")
    asyncio.run(connector.connect(output_interface))

    assert fake_serial.writes == [b"actuator:2\r\n"]


def test_connect_skips_write_when_serial_not_open(serial_connector_module):
    config = serial_connector_module.MoveSerialConfig(port="")
    connector = serial_connector_module.MoveSerialConnector(config)
    fake_serial = FakeSerial(is_open=False)
    connector.ser = fake_serial

    output_interface = types.SimpleNamespace(action="small jump")
    asyncio.run(connector.connect(output_interface))

    assert fake_serial.writes == []


def test_connect_uses_empty_move_code_for_unknown_action(serial_connector_module):
    config = serial_connector_module.MoveSerialConfig(port="")
    connector = serial_connector_module.MoveSerialConnector(config)
    fake_serial = FakeSerial(is_open=True)
    connector.ser = fake_serial

    output_interface = types.SimpleNamespace(action="spin")
    asyncio.run(connector.connect(output_interface))

    assert fake_serial.writes == [b"actuator:\r\n"]
