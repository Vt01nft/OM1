import importlib
import json
import sys
import types

import pytest


@pytest.fixture
def remote_module(monkeypatch):
    """Import the remote connector module with external dependencies stubbed."""

    ws_module = types.ModuleType("om1_utils.ws")

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        def register_message_callback(self, callback):
            self.callback = callback

        def start(self):
            pass

    ws_module.Client = StubClient

    om1_utils_module = types.ModuleType("om1_utils")
    om1_utils_module.ws = ws_module
    monkeypatch.setitem(sys.modules, "om1_utils", om1_utils_module)
    monkeypatch.setitem(sys.modules, "om1_utils.ws", ws_module)

    pydantic_module = types.ModuleType("pydantic")

    class StubBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    pydantic_module.BaseModel = StubBaseModel
    pydantic_module.ConfigDict = lambda **kwargs: kwargs
    pydantic_module.Field = lambda default=None, description="": default
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_module)

    providers_module = types.ModuleType("providers")

    class StubCommandStatus:
        def __init__(self, vx, vy, vyaw, timestamp):
            self.vx = vx
            self.vy = vy
            self.vyaw = vyaw
            self.timestamp = timestamp

        @classmethod
        def from_dict(cls, data):
            return cls(
                vx=data.get("vx", 0.0),
                vy=data.get("vy", 0.0),
                vyaw=data.get("vyaw", 0.0),
                timestamp=data.get("timestamp", 0.0),
            )

        def to_dict(self):
            return {
                "vx": self.vx,
                "vy": self.vy,
                "vyaw": self.vyaw,
                "timestamp": self.timestamp,
            }

    providers_module.CommandStatus = StubCommandStatus
    monkeypatch.setitem(sys.modules, "providers", providers_module)

    unitree_state_module = types.ModuleType("providers.unitree_go2_state_provider")

    class StubUnitreeGo2StateProvider:
        def __init__(self):
            self.state_code = 0
            self.action_progress = 0

    unitree_state_module.UnitreeGo2StateProvider = StubUnitreeGo2StateProvider
    monkeypatch.setitem(
        sys.modules, "providers.unitree_go2_state_provider", unitree_state_module
    )

    sport_module = types.ModuleType("unitree.unitree_sdk2py.go2.sport.sport_client")

    class StubSportClient:
        def SetTimeout(self, timeout):
            self.timeout = timeout

        def Init(self):
            pass

    sport_module.SportClient = StubSportClient
    monkeypatch.setitem(
        sys.modules, "unitree.unitree_sdk2py.go2.sport.sport_client", sport_module
    )

    return importlib.import_module("actions.move_go2_teleops.connector.remote")


class FakeSportClient:
    def __init__(self):
        self.balance_stand_calls = 0
        self.moves = []

    def BalanceStand(self):
        self.balance_stand_calls += 1

    def Move(self, vx, vy, vyaw):
        self.moves.append((vx, vy, vyaw))


class FakeStateProvider:
    def __init__(self, state_code=0, action_progress=0):
        self.state_code = state_code
        self.action_progress = action_progress


def build_connector(remote_module, sport_client, state_code=0, action_progress=0):
    connector = remote_module.MoveGo2RemoteConnector.__new__(
        remote_module.MoveGo2RemoteConnector
    )
    connector.sport_client = sport_client
    connector.unitree_state_provider = FakeStateProvider(state_code, action_progress)
    return connector


def test_on_message_returns_when_sport_client_missing(remote_module):
    connector = build_connector(remote_module, sport_client=None)

    connector._on_message(json.dumps({"vx": 1.0, "vy": 0.0, "vyaw": 0.0}))


def test_on_message_skips_move_when_action_in_progress(remote_module):
    fake_sport = FakeSportClient()
    connector = build_connector(
        remote_module, sport_client=fake_sport, state_code=0, action_progress=1
    )

    connector._on_message(json.dumps({"vx": 1.0, "vy": 2.0, "vyaw": 3.0}))

    assert fake_sport.moves == []


def test_on_message_balances_and_moves_when_state_and_message_valid(remote_module):
    fake_sport = FakeSportClient()
    connector = build_connector(
        remote_module, sport_client=fake_sport, state_code=1002, action_progress=0
    )

    connector._on_message(
        json.dumps({"vx": 1.25, "vy": -0.5, "vyaw": 0.75, "timestamp": "10.0"})
    )

    assert fake_sport.balance_stand_calls == 1
    assert fake_sport.moves == [(1.25, -0.5, 0.75)]


def test_on_message_handles_invalid_json_without_moving(remote_module):
    fake_sport = FakeSportClient()
    connector = build_connector(remote_module, sport_client=fake_sport)

    connector._on_message("not-json")

    assert fake_sport.moves == []
