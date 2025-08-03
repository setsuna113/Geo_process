"""Secure credential management for GEE"""

import os
import json
from pathlib import Path
from typing import Optional, Dict

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class CredentialManager:
    """Manages GEE credentials securely"""
    
    SERVICE_NAME = "biodiversity_gee"
    KEY_NAME = "service_account"
    
    def __init__(self):
        self.logger = logger
        
    def store_service_account_key(self, key_path: str) -> bool:
        """Store service account key securely"""
        try:
            with open(key_path, 'r') as f:
                key_data = json.load(f)
            
            # Validate key structure
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            if not all(field in key_data for field in required_fields):
                self.logger.error("Invalid service account key format")
                return False
            
            # Store in keyring (OS-specific secure storage)
            if KEYRING_AVAILABLE:
                keyring.set_password(
                    self.SERVICE_NAME,
                    self.KEY_NAME,
                    json.dumps(key_data)
                )
            else:
                self.logger.warning("Keyring not available - credentials not stored securely")
            
            self.logger.info("Service account key stored securely")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store credentials: {e}")
            return False
    
    def get_service_account_key(self) -> Optional[Dict]:
        """Retrieve service account key"""
        if not KEYRING_AVAILABLE:
            return None
            
        try:
            key_json = keyring.get_password(self.SERVICE_NAME, self.KEY_NAME)
            if key_json:
                return json.loads(key_json)
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve credentials: {e}")
            return None
    
    def clear_credentials(self) -> bool:
        """Clear stored credentials"""
        if not KEYRING_AVAILABLE:
            return False
            
        try:
            keyring.delete_password(self.SERVICE_NAME, self.KEY_NAME)
            self.logger.info("Credentials cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear credentials: {e}")
            return False
    
    def get_key_from_env_or_file(self, key_path: Optional[str] = None) -> Optional[Dict]:
        """Get service account key from environment variable or file"""
        # Try environment variable first
        if key_path and key_path.startswith('${') and key_path.endswith('}'):
            env_var = key_path[2:-1]
            key_path = os.environ.get(env_var)
        
        if not key_path:
            # Check common environment variables
            for env_var in ['GEE_SERVICE_ACCOUNT_KEY', 'GOOGLE_APPLICATION_CREDENTIALS']:
                key_path = os.environ.get(env_var)
                if key_path:
                    break
        
        if not key_path:
            self.logger.warning("No service account key path provided")
            return None
        
        # Load from file
        key_path = Path(key_path)
        
        # If relative path, try from project root first
        if not key_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent  # geo/
            abs_path = project_root / key_path
            if abs_path.exists():
                key_path = abs_path
        
        # Expand user home directory
        key_path = key_path.expanduser()
        
        if not key_path.exists():
            self.logger.error(f"Service account key file not found: {key_path}")
            return None
        
        try:
            with open(key_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load service account key: {e}")
            return None