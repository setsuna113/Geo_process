"""GEE Authentication Module

Refactored from climate_gee with improvements for single account support.
"""

from typing import Optional, Dict
import time

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation, retry_with_logging
from .credentials import CredentialManager

logger = get_logger(__name__)


class GEEAuthenticator:
    """Handles Google Earth Engine authentication"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize authenticator.
        
        Args:
            config: Authentication configuration
        """
        self.logger = logger
        self.config = config or {}
        self.credentials = CredentialManager()
        self._authenticated = False
        self._project_id = None
        
    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated"""
        return self._authenticated
    
    @property
    def project_id(self) -> Optional[str]:
        """Get authenticated project ID"""
        return self._project_id
    
    @log_operation("gee_authenticate")
    @retry_with_logging(max_attempts=3, delay=2.0)
    def authenticate(self, 
                    method: Optional[str] = None,
                    service_account_key: Optional[str] = None,
                    project_id: Optional[str] = None) -> bool:
        """
        Authenticate with Google Earth Engine.
        
        Args:
            method: Authentication method ('service_account', 'user', 'default')
            service_account_key: Path to service account key file
            project_id: GCP project ID
            
        Returns:
            bool: True if authentication successful
        """
        # Use provided values or fall back to config
        method = method or self.config.get('auth_method', 'service_account')
        service_account_key = service_account_key or self.config.get('service_account_key')
        project_id = project_id or self.config.get('project_id')
        
        try:
            if method == 'service_account':
                return self._auth_service_account(service_account_key, project_id)
            elif method == 'user':
                return self._auth_user(project_id)
            elif method == 'default':
                return self._auth_default(project_id)
            else:
                self.logger.error(f"Unknown authentication method: {method}")
                raise ValueError(f"Unknown authentication method: {method}")
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise RuntimeError(f"Authentication failed: {e}")
    
    def _auth_service_account(self, key_path: Optional[str], project_id: Optional[str]) -> bool:
        """Authenticate using service account"""
        # Get credentials
        key_data = self.credentials.get_key_from_env_or_file(key_path)
        if not key_data:
            # Try from secure storage
            key_data = self.credentials.get_service_account_key()
        
        if not key_data:
            self.logger.error("No service account key available")
            return False
        
        try:
            import ee
            
            # Create credentials
            import json
            credentials = ee.ServiceAccountCredentials(
                key_data['client_email'],
                key_data=json.dumps(key_data)
            )
            
            # Initialize EE
            ee.Initialize(
                credentials=credentials,
                project=project_id or key_data.get('project_id')
            )
            
            self._authenticated = True
            self._project_id = project_id or key_data.get('project_id')
            
            # Test connection
            if self._test_connection():
                self.logger.info(f"✅ Authenticated as service account: {key_data['client_email']}")
                return True
            else:
                self._authenticated = False
                return False
                
        except Exception as e:
            self.logger.error(f"Service account authentication failed: {e}")
            self._authenticated = False
            return False
    
    def _auth_user(self, project_id: Optional[str]) -> bool:
        """Authenticate as user (interactive)"""
        try:
            import ee
            
            # This will open a browser for authentication
            ee.Authenticate()
            ee.Initialize(project=project_id)
            
            self._authenticated = True
            self._project_id = project_id
            
            if self._test_connection():
                self.logger.info("✅ Authenticated as user")
                return True
            else:
                self._authenticated = False
                return False
                
        except Exception as e:
            self.logger.error(f"User authentication failed: {e}")
            self._authenticated = False
            return False
    
    def _auth_default(self, project_id: Optional[str]) -> bool:
        """Use default credentials (e.g., from gcloud)"""
        try:
            import ee
            
            ee.Initialize(project=project_id)
            
            self._authenticated = True
            self._project_id = project_id
            
            if self._test_connection():
                self.logger.info("✅ Authenticated with default credentials")
                return True
            else:
                self._authenticated = False
                return False
                
        except Exception as e:
            self.logger.error(f"Default authentication failed: {e}")
            self._authenticated = False
            return False
    
    def _test_connection(self) -> bool:
        """Test GEE connection by making a simple request"""
        try:
            import ee
            
            # Simple test: check if we can access EE
            test_value = ee.Number(1).getInfo()
            info = {'test': test_value}
            
            if test_value == 1:
                self.logger.debug("GEE connection test successful")
                return True
            else:
                self.logger.error("GEE connection test failed: unexpected response")
                return False
                
        except Exception as e:
            self.logger.error(f"GEE connection test failed: {e}")
            return False
    
    def get_ee_module(self):
        """Get authenticated Earth Engine module"""
        if not self._authenticated:
            raise RuntimeError("Not authenticated with GEE")
        import ee
        return ee
    
    def close(self):
        """Clean up authentication"""
        self._authenticated = False
        self._project_id = None
        self.logger.debug("GEE authentication closed")