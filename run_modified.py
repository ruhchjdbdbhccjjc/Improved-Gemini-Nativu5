#!/usr/bin/env python3
"""
Run the modified Gemini-FastAPI with image generation support
Supports multiple accounts from accounts.json with availability tracking
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from loguru import logger

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.main import create_app
from app.utils.cookie_loader import AccountManager, load_cookies_from_file, update_config_with_cookies
from auto_cookie_refresh import CookieRefreshManager
from validate_cookies import validate_gemini_cookies
import uvicorn


def is_account_available_now(account: Dict) -> bool:
    """
    Check if an account is currently available (not time-blocked).
    
    Args:
        account: Account dictionary from accounts.json
        
    Returns:
        True if account is available now, False otherwise
    """
    # Check if account is marked as available
    if not account.get('available', True):
        unavailable_until = account.get('unavailable_until')
        if unavailable_until:
            try:
                if isinstance(unavailable_until, str):
                    until_time = datetime.fromisoformat(unavailable_until)
                else:
                    until_time = unavailable_until
                
                if datetime.now() < until_time:
                    return False  # Still unavailable
                else:
                    # Time expired, account should be available again
                    return True
            except Exception:
                # If we can't parse the time, assume unavailable
                return False
        return False  # Marked as unavailable with no expiry
    
    return True  # Account is available


async def validate_cookies_from_account(account: Dict) -> bool:
    """
    Validate cookies from accounts.json by testing with actual GeminiClient initialization.
    This is more accurate than HTTP checks - it tests if cookies work for API calls.
    
    Returns:
        True if cookies are valid, False otherwise
    """
    secure_1psid = account.get('secure_1psid', '')
    secure_1psidts = account.get('secure_1psidts', '')
    
    if not secure_1psid or not secure_1psidts:
        return False
    
    try:
        from gemini_webapi import GeminiClient
        from gemini_webapi.exceptions import AuthError
        
        client = GeminiClient(
            secure_1psid=secure_1psid,
            secure_1psidts=secure_1psidts
        )
        
        await client.init(
            timeout=15,
            auto_close=False,
            auto_refresh=False,
            verbose=False
        )
        
        await client.close()
        return True
        
    except AuthError:
        return False
    except Exception:
        return False


async def refresh_account_cookies(account: Dict, account_manager: AccountManager, force_refresh: bool = False) -> bool:
    """
    Refresh cookies for a single account.
    
    Args:
        account: Account dictionary
        account_manager: AccountManager instance
        force_refresh: If True, refresh even if cookies are valid
        
    Returns:
        True if cookies were refreshed successfully, False otherwise
    """
    client_id = account.get('client_id')
    email = account.get('email')
    password = account.get('password')
    browser_profile_dir = account.get('browser_profile_dir', f"browser-profile-{client_id}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"🔄 Processing Account: {client_id}")
    logger.info(f"   Email: {email}")
    logger.info(f"   Enabled: {account.get('enabled', True)}")
    logger.info(f"   Browser Profile: {browser_profile_dir}")
    logger.info("=" * 60)
    
    # ✅ ONLY CHECK: Account is ENABLED (hard requirement)
    # Note: We ignore 'available' field at startup because it changes during runtime
    # 'available' is only used during runtime to track rate limits
    if not account.get('enabled', True):
        logger.info(f"⏭️  {client_id}: Skipping - account is disabled (enabled: false)")
        return False
    
    # Check if cookies exist and are valid (unless force refresh)
    if not force_refresh:
        secure_1psid = account.get('secure_1psid', '')
        secure_1psidts = account.get('secure_1psidts', '')
        
        if secure_1psid and secure_1psidts:
            logger.info(f"🔍 {client_id}: Validating existing cookies with GeminiClient...")
            is_valid = await validate_cookies_from_account(account)
            
            if is_valid:
                logger.success(f"✅ {client_id}: Existing cookies are valid for API use!")
                logger.info(f"   No refresh needed")
                
                # ✅ Ensure account is marked as available if cookies are valid
                account = account_manager.get_account(client_id, enabled_only=False)
                if account and not account.get('available', True):
                    account['available'] = True
                    account['unavailable_until'] = None
                    account['unavailable_reason'] = None
                    account_manager._save_accounts()
                    logger.info(f"   ✅ Account marked as available (cookies are valid)")
                
                return True
            else:
                logger.warning(f"⚠️ {client_id}: Existing cookies are invalid/expired for API use")
                logger.info(f"   🔄 Will attempt to refresh cookies via Puppeteer...")
    
    # Create a temporary cookie file for this account
    temp_cookie_file = Path(f"cookies_temp_{client_id}.txt")
    
    try:
        # Create CookieRefreshManager for this account
        manager = CookieRefreshManager(
            email=email,
            password=password,
            cookie_file=str(temp_cookie_file),
            max_age_hours=24,
            browser_profile_dir=browser_profile_dir,
            debug_mode=False
        )
        
        logger.info(f"🔄 {client_id}: Refreshing cookies via Puppeteer...")
        success = await manager.refresh_cookies()
        
        if success and temp_cookie_file.exists():
            # Load cookies from temp file
            secure_1psid, secure_1psidts = load_cookies_from_file(str(temp_cookie_file))
            
            if secure_1psid and secure_1psidts:
                # Update accounts.json with fresh cookies
                account_manager.update_account_cookies(client_id, secure_1psid, secure_1psidts)
                account = account_manager.get_account(client_id, enabled_only=False)
                logger.success(f"✅ {client_id}: New cookies obtained from Puppeteer!")
                
                # Validate the new cookies before marking as available
                logger.info(f"   🔍 Validating new cookies...")
                is_valid = await validate_cookies_from_account(account)
                
                if is_valid:
                    logger.success(f"   ✅ {client_id}: New cookies are valid!")
                    account['available'] = True
                    account['unavailable_until'] = None
                    account['unavailable_reason'] = None
                    account_manager._save_accounts()
                    logger.success(f"   Account marked as available")
                    
                    # Clean up temp file
                    temp_cookie_file.unlink()
                    return True
                else:
                    logger.warning(f"   ❌ {client_id}: New cookies are invalid - marking unavailable")
                    from datetime import timedelta
                    account['available'] = False
                    account['unavailable_until'] = (datetime.now() + timedelta(hours=1)).isoformat()
                    account['unavailable_reason'] = 'invalid_cookies'
                    account_manager._save_accounts()
                    return False
            else:
                logger.error(f"❌ {client_id}: Failed to extract cookies from Puppeteer")
                return False
        else:
            logger.error(f"❌ {client_id}: Puppeteer cookie refresh failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ {client_id}: Error during refresh: {e}")
        return False
    finally:
        # Clean up temp file if it exists
        if temp_cookie_file.exists():
            temp_cookie_file.unlink()


async def check_all_accounts_availability():
    """
    First run: Check all enabled accounts for availability (validate cookies).
    If cookies are invalid, attempt to refresh them using Puppeteer browser.
    Then validate again and mark as available or unavailable.
    
    Returns:
        Number of available accounts found
    """
    base_dir = Path(__file__).parent
    accounts_file = base_dir / "accounts.json"
    
    # Check if accounts.json exists
    if not accounts_file.exists():
        logger.warning("⚠️ accounts.json not found, falling back to single-account mode")
        return 0
    
    # Load account manager
    account_manager = AccountManager(str(accounts_file))
    
    # Get all accounts
    all_accounts = account_manager.get_accounts(enabled_only=False)
    enabled_accounts = account_manager.get_accounts(enabled_only=True)
    
    logger.info("=" * 60)
    logger.info("🔍 Checking Account Availability (First Run)")
    logger.info("=" * 60)
    logger.info("")
    
    # Show account status
    logger.info(f"📋 Total accounts: {len(all_accounts)}")
    logger.info(f"   Enabled: {len(enabled_accounts)}")
    logger.info(f"   Disabled: {len(all_accounts) - len(enabled_accounts)}")
    logger.info("")
    
    if not enabled_accounts:
        logger.error("❌ No enabled accounts found in accounts.json!")
        return 0
    
    # Check availability for all enabled accounts
    available_count = 0
    unavailable_count = 0
    
    for i, account in enumerate(enabled_accounts, 1):
        client_id = account.get('client_id')
        email = account.get('email')
        password = account.get('password')
        browser_profile_dir = account.get('browser_profile_dir', f"browser-profile-{client_id}")
        
        logger.info(f"[{i}/{len(enabled_accounts)}] Checking {client_id}...")
        
        # Validate cookies
        secure_1psid = account.get('secure_1psid', '')
        secure_1psidts = account.get('secure_1psidts', '')
        
        if not secure_1psid or not secure_1psidts:
            logger.warning(f"   ⚠️ {client_id}: No cookies found")
            logger.info(f"   🔄 Attempting to get new cookies via Puppeteer...")
            
            # Try to get new cookies
            temp_cookie_file = Path(f"cookies_temp_{client_id}.txt")
            try:
                manager = CookieRefreshManager(
                    email=email,
                    password=password,
                    cookie_file=str(temp_cookie_file),
                    max_age_hours=24,
                    browser_profile_dir=browser_profile_dir,
                    debug_mode=False
                )
                
                success = await manager.refresh_cookies()
                
                if success and temp_cookie_file.exists():
                    secure_1psid, secure_1psidts = load_cookies_from_file(str(temp_cookie_file))
                    
                    if secure_1psid and secure_1psidts:
                        # Update account with new cookies
                        account_manager.update_account_cookies(client_id, secure_1psid, secure_1psidts)
                        account = account_manager.get_account(client_id, enabled_only=False)
                        logger.success(f"   ✅ {client_id}: New cookies obtained!")
                        
                        # Validate new cookies
                        logger.info(f"   🔍 Validating new cookies...")
                        is_valid = await validate_cookies_from_account(account)
                        
                        if is_valid:
                            logger.success(f"   ✅ {client_id}: New cookies are valid - AVAILABLE")
                            account['available'] = True
                            account['unavailable_until'] = None
                            account['unavailable_reason'] = None
                            account_manager._save_accounts()
                            available_count += 1
                        else:
                            logger.warning(f"   ❌ {client_id}: New cookies are invalid - UNAVAILABLE")
                            from datetime import timedelta
                            account['available'] = False
                            account['unavailable_until'] = (datetime.now() + timedelta(hours=1)).isoformat()
                            account['unavailable_reason'] = 'invalid_cookies'
                            account_manager._save_accounts()
                            unavailable_count += 1
                    else:
                        logger.error(f"   ❌ {client_id}: Failed to extract cookies")
                        account['available'] = False
                        account['unavailable_reason'] = 'no_cookies'
                        account['unavailable_until'] = None
                        account_manager._save_accounts()
                        unavailable_count += 1
                else:
                    logger.error(f"   ❌ {client_id}: Failed to get new cookies")
                    account['available'] = False
                    account['unavailable_reason'] = 'no_cookies'
                    account['unavailable_until'] = None
                    account_manager._save_accounts()
                    unavailable_count += 1
            except Exception as e:
                logger.error(f"   ❌ {client_id}: Error getting cookies: {e}")
                account['available'] = False
                account['unavailable_reason'] = 'no_cookies'
                account['unavailable_until'] = None
                account_manager._save_accounts()
                unavailable_count += 1
            finally:
                if temp_cookie_file.exists():
                    temp_cookie_file.unlink()
            
            # Small delay between checks
            if i < len(enabled_accounts):
                await asyncio.sleep(2)
            continue
        
        # Test if cookies are valid
        logger.info(f"   🔍 Validating cookies...")
        is_valid = await validate_cookies_from_account(account)
        
        if is_valid:
            logger.success(f"   ✅ {client_id}: Cookies are valid - AVAILABLE")
            account['available'] = True
            account['unavailable_until'] = None
            account['unavailable_reason'] = None
            account_manager._save_accounts()
            available_count += 1
        else:
            logger.warning(f"   ❌ {client_id}: Cookies are invalid")
            logger.info(f"   🔄 Attempting to refresh cookies via Puppeteer...")
            
            # Try to refresh cookies
            temp_cookie_file = Path(f"cookies_temp_{client_id}.txt")
            try:
                manager = CookieRefreshManager(
                    email=email,
                    password=password,
                    cookie_file=str(temp_cookie_file),
                    max_age_hours=24,
                    browser_profile_dir=browser_profile_dir,
                    debug_mode=False
                )
                
                success = await manager.refresh_cookies()
                
                if success and temp_cookie_file.exists():
                    secure_1psid, secure_1psidts = load_cookies_from_file(str(temp_cookie_file))
                    
                    if secure_1psid and secure_1psidts:
                        # Update account with new cookies
                        account_manager.update_account_cookies(client_id, secure_1psid, secure_1psidts)
                        account = account_manager.get_account(client_id, enabled_only=False)
                        logger.success(f"   ✅ {client_id}: Cookies refreshed!")
                        
                        # Validate refreshed cookies
                        logger.info(f"   🔍 Validating refreshed cookies...")
                        is_valid = await validate_cookies_from_account(account)
                        
                        if is_valid:
                            logger.success(f"   ✅ {client_id}: Refreshed cookies are valid - AVAILABLE")
                            account['available'] = True
                            account['unavailable_until'] = None
                            account['unavailable_reason'] = None
                            account_manager._save_accounts()
                            available_count += 1
                        else:
                            logger.warning(f"   ❌ {client_id}: Refreshed cookies are invalid - UNAVAILABLE")
                            from datetime import timedelta
                            account['available'] = False
                            account['unavailable_until'] = (datetime.now() + timedelta(hours=1)).isoformat()
                            account['unavailable_reason'] = 'invalid_cookies'
                            account_manager._save_accounts()
                            unavailable_count += 1
                    else:
                        logger.error(f"   ❌ {client_id}: Failed to extract refreshed cookies")
                        from datetime import timedelta
                        account['available'] = False
                        account['unavailable_until'] = (datetime.now() + timedelta(hours=1)).isoformat()
                        account['unavailable_reason'] = 'invalid_cookies'
                        account_manager._save_accounts()
                        unavailable_count += 1
                else:
                    logger.error(f"   ❌ {client_id}: Failed to refresh cookies")
                    from datetime import timedelta
                    account['available'] = False
                    account['unavailable_until'] = (datetime.now() + timedelta(hours=1)).isoformat()
                    account['unavailable_reason'] = 'invalid_cookies'
                    account_manager._save_accounts()
                    unavailable_count += 1
            except Exception as e:
                logger.error(f"   ❌ {client_id}: Error refreshing cookies: {e}")
                from datetime import timedelta
                account['available'] = False
                account['unavailable_until'] = (datetime.now() + timedelta(hours=1)).isoformat()
                account['unavailable_reason'] = 'invalid_cookies'
                account_manager._save_accounts()
                unavailable_count += 1
            finally:
                if temp_cookie_file.exists():
                    temp_cookie_file.unlink()
        
        # Small delay between checks
        if i < len(enabled_accounts):
            await asyncio.sleep(2)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"📊 Availability Check Summary:")
    logger.info(f"   Available: {available_count}/{len(enabled_accounts)}")
    logger.info(f"   Unavailable: {unavailable_count}/{len(enabled_accounts)}")
    logger.info("=" * 60)
    logger.info("")
    
    return available_count


async def check_and_refresh_all_accounts():
    """
    First run: Check all enabled accounts for availability.
    Only processes available accounts.
    
    Returns:
        Number of accounts successfully processed
    """
    base_dir = Path(__file__).parent
    accounts_file = base_dir / "accounts.json"
    config_file = base_dir / "config" / "config.yaml"
    
    # Check if accounts.json exists
    if not accounts_file.exists():
        logger.warning("⚠️ accounts.json not found, falling back to single-account mode")
        return await check_and_refresh_single_account()
    
    # First: Check availability of all accounts
    available_count = await check_all_accounts_availability()
    
    # Load account manager again (after availability check)
    account_manager = AccountManager(str(accounts_file))
    enabled_accounts = account_manager.get_accounts(enabled_only=True)
    
    # Get only available accounts
    available_accounts = [
        acc for acc in enabled_accounts 
        if is_account_available_now(acc)
    ]
    
    logger.info("=" * 60)
    logger.info("🍪 Processing Available Accounts")
    logger.info("=" * 60)
    logger.info("")
    
    if not available_accounts:
        logger.warning("⚠️ No available accounts to process!")
        return 0
    
    logger.info(f"📋 Processing {len(available_accounts)} available account(s)")
    logger.info("")
    
    # Process only available accounts
    success_count = 0
    
    for i, account in enumerate(available_accounts, 1):
        client_id = account.get('client_id')
        
        logger.info(f"[{i}/{len(available_accounts)}] Processing {client_id}...")
        
        # Refresh cookies for available accounts
        success = await refresh_account_cookies(account, account_manager, force_refresh=False)
        
        if success:
            success_count += 1
            logger.success(f"✅ {client_id}: Completed successfully")
        else:
            logger.warning(f"⚠️ {client_id}: Failed or skipped")
        
        # Small delay between accounts
        if i < len(available_accounts):
            logger.info("⏳ Waiting 3 seconds before next account...")
            await asyncio.sleep(3)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"📊 Summary: {success_count}/{len(available_accounts)} available account(s) processed successfully")
    logger.info("=" * 60)
    
    # Update config.yaml with available accounts
    if success_count > 0:
        logger.info("")
        logger.info("📝 Updating config.yaml with available accounts...")
        account_manager.update_config_from_accounts(str(config_file))
        logger.success("✅ Config updated!")
    
    return success_count


async def check_and_refresh_single_account():
    """
    Fallback: Check and refresh cookies for single account mode (cookies.txt).
    
    Returns:
        1 if successful, 0 otherwise
    """
    logger.info("🍪 Single-Account Cookie Refresh Mode")
    logger.info("=" * 60)
    
    cookie_file = Path(__file__).parent / "cookies.txt"
    config_file = Path(__file__).parent / "config" / "config.yaml"
    
    # Load cookies from file
    if cookie_file.exists():
        secure_1psid, secure_1psidts = load_cookies_from_file(str(cookie_file))
        
        if secure_1psid and secure_1psidts:
            logger.info(f"✅ Found cookies in {cookie_file.name}")
            logger.info(f"   __Secure-1PSID: {secure_1psid[:30]}...")
            logger.info(f"   __Secure-1PSIDTS: {secure_1psidts[:30]}...")
            
            # Validate cookies
            logger.info("🔍 Validating cookies...")
            is_valid = await validate_gemini_cookies(secure_1psid, secure_1psidts)
            
            if is_valid:
                logger.success("✅ Cookies are valid!")
                logger.info("📝 Updating config.yaml...")
                update_config_with_cookies(str(config_file), str(cookie_file))
                logger.success("✅ Config updated!")
                return 1
            else:
                logger.warning("❌ Cookies are invalid or rate limited!")
                logger.info("🔄 Attempting to refresh cookies...")
                
                # Try to refresh cookies
                email = os.getenv('GEMINI_EMAIL', 'trungthuphan160@gmail.com')
                password = os.getenv('GEMINI_PASSWORD', '123456.*@a')
                
                manager = CookieRefreshManager(
                    email=email,
                    password=password,
                    cookie_file='cookies.txt',
                    max_age_hours=24,
                    debug_mode=False
                )
                
                success = await manager.refresh_cookies()
                
                if success:
                    logger.success("✅ Cookie refresh successful!")
                    update_config_with_cookies(str(config_file), str(cookie_file))
                    return 1
                else:
                    logger.error("❌ Cookie refresh failed!")
                    return 0
        else:
            logger.error("❌ Could not extract cookies from file")
            return 0
    else:
        logger.warning(f"❌ Cookie file not found: {cookie_file}")
        logger.info("🔄 Creating cookies...")
        
        email = os.getenv('GEMINI_EMAIL', 'trungthuphan160@gmail.com')
        password = os.getenv('GEMINI_PASSWORD', '123456.*@a')
        
        manager = CookieRefreshManager(
            email=email,
            password=password,
            cookie_file='cookies.txt',
            debug_mode=False
        )
        
        success = await manager.refresh_cookies()
        
        if success:
            logger.success("✅ Cookies created!")
            update_config_with_cookies(str(config_file), str(cookie_file))
            return 1
        else:
            logger.error("❌ Failed to create cookies!")
            return 0


def main():
    """Main function to run the server"""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("Multi-Account Support with Availability Tracking")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🔄 Startup Sequence:")
    logger.info("   1️⃣  Check all accounts for availability")
    logger.info("   2️⃣  Process only available accounts")
    logger.info("")
    
    # Check and refresh cookies for all accounts
    success_count = asyncio.run(check_and_refresh_all_accounts())
    
    if success_count == 0:
        logger.warning("")
        logger.warning("=" * 60)
        logger.warning("⚠️  WARNING: No accounts were refreshed successfully")
        logger.warning("=" * 60)
        logger.warning("")
        logger.warning("Possible issues:")
        logger.warning("  1. All accounts are unavailable (rate limited, invalid cookies)")
        logger.warning("  2. Accounts are disabled (check accounts.json)")
        logger.warning("  3. Network issues")
        logger.warning("")
        logger.warning("The server will start with existing cookies if available.")
        logger.warning("Unavailable accounts will be skipped during requests.")
        logger.warning("")
        logger.warning("Try:")
        logger.warning("  python validate_cookies.py  # Check cookie status")
        logger.warning("  python run_with_auto_refresh.py  # Force refresh all accounts")
        logger.warning("=" * 60)
        logger.warning("")
    else:
        logger.success(f"✅ Successfully refreshed {success_count} account(s)")
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Gemini-FastAPI Server")
    logger.info("=" * 60)
    logger.info("")
    logger.info("API Documentation: http://localhost:30000/docs")
    logger.info("Image Generation: POST /v1/images/generations")
    logger.info("Chat Completion: POST /v1/chat/completions")
    logger.info("Native Gemini API: POST /v1beta/models/generateContent")
    logger.info("Model-specific API: POST /v1beta/models/{model}:generateContent")
    logger.info("")
    logger.info("=" * 60)
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=30000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
