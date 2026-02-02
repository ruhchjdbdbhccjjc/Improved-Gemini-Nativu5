"""
Cookie loader utility to automatically load fresh cookies from file
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def parse_cookie_string(cookie_string: str) -> Dict[str, str]:
    """
    Parse a cookie string and return a dictionary of key-value pairs.
    
    Args:
        cookie_string: Raw cookie string from browser
        
    Returns:
        Dictionary of cookie key-value pairs
    """
    cookies = {}
    
    # Split by semicolon and parse each cookie
    for cookie in cookie_string.split(';'):
        cookie = cookie.strip()
        if '=' in cookie:
            key, value = cookie.split('=', 1)
            cookies[key.strip()] = value.strip()
    
    return cookies


def load_cookies_from_file(cookie_file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Load cookies from file and extract the required Gemini cookies.
    
    Args:
        cookie_file_path: Path to the cookie file
        
    Returns:
        Tuple of (secure_1psid, secure_1psidts) or (None, None) if not found
    """
    try:
        cookie_path = Path(cookie_file_path)
        
        if not cookie_path.exists():
            print(f"❌ Cookie file not found: {cookie_file_path}")
            return None, None
        
        # Read the cookie file
        with open(cookie_path, 'r', encoding='utf-8') as f:
            cookie_string = f.read().strip()
        
        if not cookie_string:
            print(f"❌ Cookie file is empty: {cookie_file_path}")
            return None, None
        
        # Parse the cookie string
        cookies = parse_cookie_string(cookie_string)
        
        # Extract the required cookies
        secure_1psid = cookies.get('__Secure-1PSID')
        secure_1psidts = cookies.get('__Secure-1PSIDTS')
        
        if not secure_1psid:
            print("ERROR: __Secure-1PSID not found in cookie file")
            return None, None
        
        if not secure_1psidts:
            print("ERROR: __Secure-1PSIDTS not found in cookie file")
            return None, None
        
        print(f"SUCCESS: Successfully loaded cookies from {cookie_file_path}")
        print(f"   __Secure-1PSID: {secure_1psid[:20]}...")
        print(f"   __Secure-1PSIDTS: {secure_1psidts[:20]}...")
        
        return secure_1psid, secure_1psidts
        
    except Exception as e:
        print(f"ERROR: Error loading cookies from file {cookie_file_path}: {e}")
        return None, None


def update_config_with_cookies(config_path: str, cookie_file_path: str) -> bool:
    """
    Update the config file with fresh cookies from the cookie file.
    
    Args:
        config_path: Path to the config.yaml file
        cookie_file_path: Path to the cookies.txt file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load cookies from file
        secure_1psid, secure_1psidts = load_cookies_from_file(cookie_file_path)
        
        if not secure_1psid or not secure_1psidts:
            print("❌ Failed to load cookies from file")
            return False
        
        # Read the current config
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Update the cookies in the config
        # Replace the secure_1psid line
        secure_1psid_pattern = r'secure_1psid: ".*?"'
        secure_1psid_replacement = f'secure_1psid: "{secure_1psid}"'
        config_content = re.sub(secure_1psid_pattern, secure_1psid_replacement, config_content)
        
        # Replace the secure_1psidts line
        secure_1psidts_pattern = r'secure_1psidts: ".*?"'
        secure_1psidts_replacement = f'secure_1psidts: "{secure_1psidts}"'
        config_content = re.sub(secure_1psidts_pattern, secure_1psidts_replacement, config_content)
        
        # Write the updated config
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"SUCCESS: Successfully updated config with fresh cookies from {cookie_file_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Error updating config with cookies: {e}")
        return False


def get_cookies_from_file(cookie_file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get cookies from file (convenience function for direct use).
    
    Args:
        cookie_file_path: Path to the cookie file
        
    Returns:
        Tuple of (secure_1psid, secure_1psidts) or (None, None) if not found
    """
    return load_cookies_from_file(cookie_file_path)


class AccountManager:
    """Manages accounts from accounts.json file."""
    
    def __init__(self, accounts_file_path: str):
        """
        Initialize AccountManager with accounts.json file path.
        
        Args:
            accounts_file_path: Path to accounts.json file
        """
        self.accounts_file = Path(accounts_file_path)
        self._accounts_data = None
        self._load_accounts()
    
    def _load_accounts(self):
        """Load accounts from JSON file."""
        try:
            if not self.accounts_file.exists():
                self._accounts_data = {"accounts": []}
                return
            
            with open(self.accounts_file, 'r', encoding='utf-8') as f:
                self._accounts_data = json.load(f)
            
            if 'accounts' not in self._accounts_data:
                self._accounts_data = {"accounts": []}
        except Exception as e:
            print(f"ERROR: Failed to load accounts.json: {e}")
            self._accounts_data = {"accounts": []}
    
    def _save_accounts(self):
        """Save accounts to JSON file."""
        try:
            with open(self.accounts_file, 'w', encoding='utf-8') as f:
                json.dump(self._accounts_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"ERROR: Failed to save accounts.json: {e}")
            raise
    
    def get_accounts(self, enabled_only: bool = True) -> List[Dict]:
        """
        Get all accounts, optionally filtered by enabled status.
        
        Args:
            enabled_only: If True, only return enabled accounts
            
        Returns:
            List of account dictionaries
        """
        accounts = self._accounts_data.get('accounts', [])
        
        if enabled_only:
            return [acc for acc in accounts if acc.get('enabled', True)]
        
        return accounts
    
    def get_account(self, client_id: str, enabled_only: bool = False) -> Optional[Dict]:
        """
        Get a specific account by client_id.
        
        Args:
            client_id: The client ID to look for
            enabled_only: If True, only return if account is enabled
            
        Returns:
            Account dictionary or None if not found
        """
        accounts = self.get_accounts(enabled_only=False)
        
        for account in accounts:
            if account.get('client_id') == client_id:
                if enabled_only and not account.get('enabled', True):
                    return None
                return account
        
        return None
    
    def update_account_cookies(self, client_id: str, secure_1psid: str, secure_1psidts: str):
        """
        Update cookies for a specific account.
        
        Args:
            client_id: The client ID to update
            secure_1psid: New secure_1psid cookie value
            secure_1psidts: New secure_1psidts cookie value
        """
        account = self.get_account(client_id, enabled_only=False)
        
        if account:
            account['secure_1psid'] = secure_1psid
            account['secure_1psidts'] = secure_1psidts
            account['last_refreshed'] = datetime.now().isoformat()
            self._save_accounts()
        else:
            raise ValueError(f"Account with client_id '{client_id}' not found")
    
    def update_config_from_accounts(self, config_file_path: str):
        """
        Update config.yaml with cookies from accounts.json.
        
        Args:
            config_file_path: Path to config.yaml file
        """
        try:
            config_file = Path(config_file_path)
            
            if not config_file.exists():
                print(f"❌ Config file not found: {config_file_path}")
                return False
            
            # Load config
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Get enabled accounts
            accounts = self.get_accounts(enabled_only=True)
            
            # Update clients in config
            if 'gemini' not in config_data:
                config_data['gemini'] = {}
            
            if 'clients' not in config_data['gemini']:
                config_data['gemini']['clients'] = []
            
            # Create a mapping of client_id to account
            account_map = {acc['client_id']: acc for acc in accounts}
            
            # Update existing clients or add new ones
            existing_client_ids = {client['id'] for client in config_data['gemini']['clients']}
            
            for account in accounts:
                client_id = account['client_id']
                secure_1psid = account.get('secure_1psid', '')
                secure_1psidts = account.get('secure_1psidts', '')
                pro = account.get('pro', False)
                
                # Find existing client or create new one
                client_found = False
                for client in config_data['gemini']['clients']:
                    if client['id'] == client_id:
                        client['secure_1psid'] = secure_1psid
                        client['secure_1psidts'] = secure_1psidts
                        client['pro'] = pro
                        client_found = True
                        break
                
                if not client_found:
                    # Add new client
                    config_data['gemini']['clients'].append({
                        'id': client_id,
                        'secure_1psid': secure_1psid,
                        'secure_1psidts': secure_1psidts,
                        'pro': pro
                    })
            
            # Save updated config
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"✅ Successfully updated config.yaml with {len(accounts)} enabled account(s)")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to update config.yaml: {e}")
            return False


if __name__ == "__main__":
    # Test the cookie loader
    cookie_file = "cookies.txt"
    config_file = "config/config.yaml"
    
    print("🍪 Testing Cookie Loader")
    print("=" * 40)
    
    # Test loading cookies
    secure_1psid, secure_1psidts = load_cookies_from_file(cookie_file)
    
    if secure_1psid and secure_1psidts:
        print(f"✅ Successfully loaded cookies:")
        print(f"   __Secure-1PSID: {secure_1psid[:30]}...")
        print(f"   __Secure-1PSIDTS: {secure_1psidts[:30]}...")
        
        # Test updating config
        if update_config_with_cookies(config_file, cookie_file):
            print(f"✅ Successfully updated config file")
        else:
            print(f"❌ Failed to update config file")
    else:
        print(f"❌ Failed to load cookies from {cookie_file}")
