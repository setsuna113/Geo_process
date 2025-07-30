"""Google Earth Engine Authentication Module"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

try:
    from src.infrastructure.logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class GEEAuthenticator:
    """Handles Google Earth Engine authentication with multiple methods."""
    
    def __init__(self, logger=None):
        self.logger = logger if logger else get_logger(__name__)
        self.ee = None
        self._authenticated = False
        
    def authenticate(self, 
                    service_account_key: Optional[str] = None,
                    use_service_account: bool = True,
                    project_id: Optional[str] = None) -> bool:
        """
        Authenticate with Google Earth Engine.
        
        Args:
            service_account_key: Path to service account JSON key file
            use_service_account: Whether to use service account authentication
            project_id: GEE project ID
            
        Returns:
            bool: True if authentication successful
        """
        try:
            import ee
            self.ee = ee
            
            if use_service_account and service_account_key:
                return self._authenticate_service_account(service_account_key, project_id)
            else:
                return self._authenticate_user_account(project_id)
                
        except ImportError as e:
            self.logger.error(f"earthengine-api not installed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"GEE authentication failed: {e}")
            return False
    
    def _authenticate_service_account(self, key_file: str, project_id: Optional[str]) -> bool:
        """Authenticate using service account credentials."""
        try:
            key_path = Path(key_file)
            if not key_path.exists():
                raise FileNotFoundError(f"Service account key file not found: {key_file}")
            
            # Load and validate service account key
            with open(key_path) as f:
                credentials = json.load(f)
            
            if 'client_email' not in credentials:
                raise ValueError("Invalid service account key file")
            
            # Initialize with service account
            service_account = credentials['client_email']
            self.ee.Initialize(
                credentials=self.ee.ServiceAccountCredentials(service_account, key_file),
                project=project_id
            )
            
            self._authenticated = True
            self.logger.info(f"GEE authenticated with service account: {service_account}")
            
            return True
            
        except Exception as e:
            # Sanitize error message to avoid credential exposure
            safe_error = str(e).replace(str(key_path), '[KEY_FILE]')
            if 'credentials' in safe_error.lower():
                safe_error = "Authentication failed - check service account credentials"
            
            self.logger.error(f"Service account authentication failed: {safe_error}")
            return False
    
    def _authenticate_user_account(self, project_id: Optional[str]) -> bool:
        """Authenticate using user account (interactive)."""
        try:
            # Try to authenticate with existing token
            try:
                self.ee.Initialize(project=project_id)
                self._authenticated = True
                self.logger.info("GEE authenticated with existing user credentials")
                return True
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Existing credentials not valid, will attempt interactive auth: {e}")
                # Continue to interactive authentication
            
            # Need to authenticate interactively
            self.logger.info("Starting interactive GEE authentication...")
            
            self.ee.Authenticate()
            self.ee.Initialize(project=project_id)
            
            self._authenticated = True
            self.logger.info("GEE user authentication completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"User authentication failed: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        if not self._authenticated or not self.ee:
            return False
        
        try:
            # Test authentication with a simple operation
            self.ee.Number(1).getInfo()
            return True
        except Exception as e:
            self.logger.debug(f"Authentication test failed: {e}")
            self._authenticated = False
            return False
    
    def get_project_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current GEE project."""
        if not self.is_authenticated():
            return None
        
        try:
            # Get some basic project info
            info = {
                'authenticated': True,
                'available_assets': True  # Could expand this
            }
            
            if self.logger:
                self.logger.debug(f"GEE project info: {info}")
            
            return info
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get project info: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test the GEE connection with a simple operation."""
        if not self.is_authenticated():
            return False
        
        try:
            # Simple test: get image info
            image = self.ee.Image("WORLDCLIM/V1/BIO").select("bio01")
            info = image.getInfo()
            
            self.logger.info("GEE connection test successful")
            
            return True
            
        except Exception as e:
            self.logger.error(f"GEE connection test failed: {e}")
            return False


def setup_gee_auth(service_account_key: Optional[str] = None,
                  project_id: Optional[str] = None,
                  logger=None) -> Optional[GEEAuthenticator]:
    """
    Convenience function to set up GEE authentication.
    
    Args:
        service_account_key: Path to service account JSON file
        project_id: GEE project ID
        logger: Logger instance
        
    Returns:
        GEEAuthenticator instance if successful, None otherwise
    """
    auth = GEEAuthenticator(logger)
    
    # Try service account first if key provided
    if service_account_key:
        if auth.authenticate(service_account_key=service_account_key, 
                           use_service_account=True, 
                           project_id=project_id):
            return auth
    
    # Fall back to user authentication
    if auth.authenticate(use_service_account=False, project_id=project_id):
        return auth
    
    return None