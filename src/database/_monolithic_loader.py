"""Helper module to load the monolithic schema.py file.

This module exists solely to work around the naming conflict between
schema.py (file) and schema/ (directory). It provides a way for the
modular schema implementation to access the monolithic implementation.
"""

import os
import types

# Cache the module to avoid reloading
_monolithic_module = None

def get_monolithic_schema_class():
    """Load and return the DatabaseSchema class from schema.py.
    
    This function explicitly loads the schema.py file (not the schema/ package)
    to get the monolithic DatabaseSchema implementation.
    """
    global _monolithic_module
    
    if _monolithic_module is None:
        # Get the path to schema.py
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.py')
        
        # Read the file
        with open(schema_path, 'r') as f:
            code = f.read()
        
        # Create a module with proper context for relative imports
        _monolithic_module = types.ModuleType('src.database.schema_monolithic')
        _monolithic_module.__file__ = schema_path
        _monolithic_module.__package__ = 'src.database'
        
        # Execute the code in the module's namespace
        exec(compile(code, schema_path, 'exec'), _monolithic_module.__dict__)
    
    return _monolithic_module.DatabaseSchema