"""Test suite for the registry system."""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from src.core.registry import Registry, ComponentRegistry, component_registry
from src.core.build import ComponentBuilder, build_grid, register_default_components

class TestRegistry:
    """Test the Registry class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.registry = Registry("test")
        
    def teardown_method(self):
        """Cleanup after each test."""
        self.registry.clear()
    
    def test_basic_registration(self):
        """Test basic class registration."""
        class TestClass:
            pass
            
        # Register class
        self.registry.register(TestClass)
        
        # Check registration
        assert "TestClass" in self.registry
        assert len(self.registry) == 1
        assert self.registry.get("TestClass") is TestClass
        
    def test_duplicate_registration_error(self):
        """Test that duplicate registration raises error."""
        class TestClass:
            pass
            
        self.registry.register(TestClass)
        
        # Try to register again
        with pytest.raises(ValueError) as exc_info:
            self.registry.register(TestClass)
            
        assert "already registered" in str(exc_info.value)
        
    def test_force_registration(self):
        """Test force registration overwrites existing."""
        class TestClass1:
            pass
            
        class TestClass2:
            pass
            
        # Rename second class to have same name
        TestClass2.__name__ = "TestClass1"
        
        self.registry.register(TestClass1)
        self.registry.register(TestClass2, force=True)
        
        # Should have the second class
        assert self.registry.get("TestClass1") is TestClass2
        
    def test_get_nonexistent_class(self):
        """Test getting non-existent class raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            self.registry.get("NonExistent")
            
        assert "not found" in str(exc_info.value)
        assert "Available: []" in str(exc_info.value)
        
    def test_decorator_registration(self):
        """Test registration via decorator."""
        @self.registry.register_decorator()
        class DecoratedClass:
            pass
            
        assert "DecoratedClass" in self.registry
        assert self.registry.get("DecoratedClass") is DecoratedClass
        
    def test_get_instance(self):
        """Test getting instances of registered classes."""
        @self.registry.register_decorator()
        class TestComponent:
            def __init__(self, value):
                self.value = value
                
        # Get instance
        instance1 = self.registry.get_instance("TestComponent", 42)
        assert instance1.value == 42
        
        # Get same instance with same args
        instance2 = self.registry.get_instance("TestComponent", 42)
        assert instance1 is instance2
        
        # Get different instance with different args
        instance3 = self.registry.get_instance("TestComponent", 100)
        assert instance3.value == 100
        assert instance3 is not instance1
        
    def test_inheritance_resolution(self):
        """Test getting classes by base class."""
        class BaseProcessor:
            pass
            
        @self.registry.register_decorator()
        class ProcessorA(BaseProcessor):
            pass
            
        @self.registry.register_decorator()
        class ProcessorB(BaseProcessor):
            pass
            
        @self.registry.register_decorator()
        class OtherClass:
            pass
            
        # Get all processors
        processors = self.registry.get_by_base_class(BaseProcessor)
        assert len(processors) == 2
        assert "ProcessorA" in processors
        assert "ProcessorB" in processors
        assert "OtherClass" not in processors
        
    def test_thread_safety(self):
        """Test concurrent registration is thread-safe."""
        results = []
        errors = []
        
        def register_class(i):
            """Register a class in a thread."""
            try:
                # Create unique class
                cls = type(f"TestClass{i}", (), {})
                self.registry.register(cls)
                results.append(i)
            except Exception as e:
                errors.append(e)
                
        # Run concurrent registrations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_class, i) for i in range(100)]
            for future in futures:
                future.result()
                
        # Check results
        assert len(errors) == 0
        assert len(results) == 100
        assert len(self.registry) == 100
        
    def test_validators(self):
        """Test class validators."""
        def require_process_method(cls):
            if not hasattr(cls, 'process'):
                raise ValueError(f"{cls.__name__} must have a 'process' method")
                
        self.registry.add_validator(require_process_method)
        
        # Valid class
        class ValidClass:
            def process(self):
                pass
                
        self.registry.register(ValidClass)  # Should work
        
        # Invalid class
        class InvalidClass:
            pass
            
        with pytest.raises(ValueError) as exc_info:
            self.registry.register(InvalidClass)
            
        assert "must have a 'process' method" in str(exc_info.value)


class TestComponentRegistry:
    """Test the ComponentRegistry class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.registry = ComponentRegistry()
        
    def teardown_method(self):
        """Cleanup after each test."""
        self.registry.grids.clear()
        self.registry.processors.clear()
        self.registry.extractors.clear()
        
    def test_grid_validation(self):
        """Test grid class validation."""
        # Invalid grid (missing methods)
        class InvalidGrid:
            pass
            
        with pytest.raises(ValueError) as exc_info:
            self.registry.grids.register(InvalidGrid)
            
        assert "missing required method" in str(exc_info.value)
        
        # Valid grid
        class ValidGrid:
            def generate_grid(self):
                pass
            def get_cell_size(self):
                pass
            def get_cell_count(self):
                pass
                
        self.registry.grids.register(ValidGrid)  # Should work
        
    def test_auto_register_module(self):
        """Test auto-registration from module."""
        # Create a mock module
        import types
        module = types.ModuleType("test_module")
        module.__name__ = "test_module"
        
        # Add classes to module
        class TestGrid:
            def generate_grid(self):
                pass
            def get_cell_size(self):
                pass
            def get_cell_count(self):
                pass
                
        class TestProcessor:
            def process_single(self):
                pass
            def process(self):
                pass
                
        TestGrid.__module__ = "test_module"
        TestProcessor.__module__ = "test_module"
        
        module.TestGrid = TestGrid
        module.TestProcessor = TestProcessor
        
        # Auto-register
        self.registry.auto_register_module(module, "grids")
        
        # Check registration
        assert "TestGrid" in self.registry.grids
        assert "TestProcessor" not in self.registry.grids  # Wrong base class


class TestComponentBuilder:
    """Test the ComponentBuilder class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.builder = ComponentBuilder()
        
        # Register test components
        @component_registry.grids.register_decorator()
        class TestGrid:
            def __init__(self, resolution=100):
                self.resolution = resolution
            def generate_grid(self):
                pass
            def get_cell_size(self):
                return self.resolution
            def get_cell_count(self):
                return 100
                
        @component_registry.processors.register_decorator()
        class TestProcessor:
            def __init__(self, batch_size=100):
                self.batch_size = batch_size
            def process_single(self, item):
                return item
            def process(self, items):
                return items
                
        self.TestGrid = TestGrid
        self.TestProcessor = TestProcessor
        
    def teardown_method(self):
        """Cleanup after each test."""
        component_registry.grids.clear()
        component_registry.processors.clear()
        
    def test_build_from_config(self):
        """Test building component from config."""
        # Build grid
        grid_config = {
            'class': 'TestGrid',
            'params': {
                'resolution': 50
            }
        }
        
        grid = self.builder.build_from_config('grid', grid_config)
        assert isinstance(grid, self.TestGrid)
        assert grid.resolution == 50
        
    def test_build_missing_class(self):
        """Test building with missing class raises error."""
        config = {
            'params': {'resolution': 50}
        }
        
        with pytest.raises(ValueError) as exc_info:
            self.builder.build_from_config('grid', config)
            
        assert "missing 'class' key" in str(exc_info.value)
        
    def test_build_nonexistent_class(self):
        """Test building non-existent class raises error."""
        config = {
            'class': 'NonExistentGrid'
        }
        
        with pytest.raises(KeyError):
            self.builder.build_from_config('grid', config)
            
    def test_build_pipeline(self):
        """Test building a pipeline of components."""
        pipeline_config = [
            {
                'type': 'grid',
                'class': 'TestGrid',
                'params': {'resolution': 25}
            },
            {
                'type': 'processor',
                'class': 'TestProcessor',
                'params': {'batch_size': 50}
            }
        ]
        
        components = self.builder.build_pipeline(pipeline_config)
        
        assert len(components) == 2
        assert isinstance(components[0], self.TestGrid)
        assert components[0].resolution == 25
        assert isinstance(components[1], self.TestProcessor)
        assert components[1].batch_size == 50
        
    def test_build_grid_helper(self):
        """Test the build_grid helper function."""
        grid_config = {
            'class': 'TestGrid',
            'params': {'resolution': 75}
        }
        
        grid = build_grid(grid_config)
        assert isinstance(grid, self.TestGrid)
        assert grid.resolution == 75


# Integration test
def test_full_integration():
    """Test full integration with actual grid classes."""
    # This would be run after grid classes are implemented
    try:
        register_default_components()
        
        # Build a cubic grid from config
        grid = build_grid({
            'grid_type': 'cubic',
            'params': {
                'resolution': 1000
            }
        })
        
        assert grid is not None
        # More assertions based on actual implementation
        
    except (ImportError, KeyError):
        pytest.skip("Grid systems not yet implemented")