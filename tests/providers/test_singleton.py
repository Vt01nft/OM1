"""Tests for singleton."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

# Mock ALL external dependencies BEFORE any provider imports
# This must happen at module load time
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["rclpy"] = MagicMock()
sys.modules["rclpy.node"] = MagicMock()
sys.modules["rclpy.qos"] = MagicMock()
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["nav_msgs.msg"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["websocket"] = MagicMock()
sys.modules["websockets"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()


class TestSingletonDecorator:
    """Tests for singleton decorator."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        # Clear cached provider modules to reset singletons
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield
        # Cleanup after test
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_singleton_decorator_creates_instance(self):
        """Test singleton decorator creates an instance correctly."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value=None):
                self.value = value

        instance = TestClass("test")
        assert instance is not None
        assert instance.value == "test"

    def test_singleton_decorator_returns_same_instance(self):
        """Test singleton decorator returns the same instance on multiple calls."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value=None):
                self.value = value

        instance1 = TestClass("first")
        instance2 = TestClass("second")

        assert instance1 is instance2
        assert instance1.value == "first"  # Should keep first initialization

    def test_singleton_decorator_with_no_args(self):
        """Test singleton decorator works with no constructor arguments."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self):
                self.created = True

        instance1 = TestClass()
        instance2 = TestClass()

        assert instance1 is instance2
        assert instance1.created is True

    def test_singleton_decorator_with_kwargs(self):
        """Test singleton decorator works with keyword arguments."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, name=None, value=None):
                self.name = name
                self.value = value

        instance1 = TestClass(name="test", value=42)
        instance2 = TestClass(name="other", value=99)

        assert instance1 is instance2
        assert instance1.name == "test"
        assert instance1.value == 42

    def test_singleton_decorator_reset_functionality(self):
        """Test singleton decorator reset method works correctly."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value=None):
                self.value = value

        instance1 = TestClass("first")
        TestClass.reset()
        instance2 = TestClass("second")

        assert instance1 is not instance2
        assert instance1.value == "first"
        assert instance2.value == "second"

    def test_singleton_decorator_thread_safety(self):
        """Test singleton decorator is thread-safe."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value=None):
                self.value = value
                # Simulate some work to increase chance of race condition
                time.sleep(0.01)

        TestClass.reset()
        instances = []

        def create_instance():
            instances.append(TestClass("test"))

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same object
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    def test_singleton_decorator_concurrent_creation(self):
        """Test singleton decorator handles concurrent instance creation."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self):
                self.thread_id = threading.current_thread().ident
                time.sleep(0.001)  # Small delay to test concurrency

        TestClass.reset()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(TestClass) for _ in range(10)]
            instances = [future.result() for future in futures]

        # All instances should be the same
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance

    def test_singleton_decorator_preserves_class_attributes(self):
        """Test singleton decorator preserves original class attributes."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            class_var = "test_value"

            def __init__(self):
                self.instance_var = "instance_value"

            def test_method(self):
                return "method_result"

        instance = TestClass()

        assert hasattr(TestClass, "_singleton_class")
        assert TestClass._singleton_class.class_var == "test_value"
        assert instance.instance_var == "instance_value"
        assert instance.test_method() == "method_result"

    def test_singleton_decorator_reset_clears_instance(self):
        """Test singleton decorator reset method clears the instance."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value=None):
                self.value = value

        # Create first instance
        instance1 = TestClass("first")
        assert instance1.value == "first"

        # Reset should clear the instance
        TestClass.reset()

        # New instance should be different
        instance2 = TestClass("second")
        assert instance2.value == "second"
        assert instance1 is not instance2

    def test_singleton_decorator_multiple_classes(self):
        """Test singleton decorator works independently for multiple classes."""
        from providers.singleton import singleton

        @singleton
        class TestClass1:
            def __init__(self, value=None):
                self.value = value

        @singleton
        class TestClass2:
            def __init__(self, value=None):
                self.value = value

        instance1a = TestClass1("class1")
        instance2a = TestClass2("class2")
        instance1b = TestClass1("class1_again")
        instance2b = TestClass2("class2_again")

        # Same class instances should be identical
        assert instance1a is instance1b
        assert instance2a is instance2b

        # Different class instances should be different
        assert instance1a is not instance2a

        # Values should be from first creation
        assert instance1a.value == "class1"
        assert instance2a.value == "class2"

    def test_singleton_decorator_reset_thread_safety(self):
        """Test singleton decorator reset method is thread-safe."""
        from providers.singleton import singleton

        @singleton
        class TestClass:
            def __init__(self, value=None):
                self.value = value

        def reset_and_create():
            TestClass.reset()
            return TestClass("thread_test")

        # Create initial instance
        initial = TestClass("initial")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(reset_and_create) for _ in range(5)]
            results = [future.result() for future in futures]

        # All operations should complete without error
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert result.value == "thread_test"
