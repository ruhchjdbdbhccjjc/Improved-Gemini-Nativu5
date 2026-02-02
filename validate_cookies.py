"""
Cookie Validation - Check if cookies are valid before using them
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import httpx


async def validate_gemini_cookies(secure_1psid: str, secure_1psidts: str) -> bool:
    """
    Validate Gemini cookies by making a test request.
    
    Args:
        secure_1psid: The __Secure-1PSID cookie value
        secure_1psidts: The __Secure-1PSIDTS cookie value
        
    Returns:
        True if cookies are valid, False otherwise
    """
    try:
        logger.info("🔍 Validating cookies by making test request to Gemini...")
        
        # Prepare cookies
        cookies = {
            '__Secure-1PSID': secure_1psid,
            '__Secure-1PSIDTS': secure_1psidts
        }
        
        # Make a simple test request to Gemini
        url = 'https://gemini.google.com/app'
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                cookies=cookies,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response URL: {response.url}")
            
            # Check if we were redirected to login
            if 'accounts.google.com' in str(response.url):
                logger.error("❌ Cookies are invalid - redirected to login page")
                return False
            
            # Check for rate limiting
            if response.status_code == 429:
                logger.warning("⚠️ Rate limited (429) - cookies might be valid but account is rate limited")
                logger.warning("   Wait 30-60 minutes before trying again")
                return False
            
            # Check for CAPTCHA/sorry page
            if 'sorry' in str(response.url) or response.status_code == 403:
                logger.error("❌ CAPTCHA/rate limit page - account is rate limited")
                logger.warning("   Wait 30-60 minutes before trying again")
                return False
            
            # Success!
            if response.status_code == 200:
                logger.success("✅ Cookies are valid! Successfully accessed Gemini!")
                return True
            
            # Unknown status
            logger.warning(f"⚠️ Unexpected response: {response.status_code}")
            logger.info("Assuming cookies might be valid...")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error validating cookies: {e}")
        return False


async def validate_cookies_from_file(cookie_file: str) -> bool:
    """
    Validate cookies from a cookie file.
    
    Args:
        cookie_file: Path to cookies.txt file
        
    Returns:
        True if cookies are valid, False otherwise
    """
    try:
        from app.utils.cookie_loader import load_cookies_from_file
        
        logger.info(f"📂 Loading cookies from {cookie_file}...")
        secure_1psid, secure_1psidts = load_cookies_from_file(cookie_file)
        
        if not secure_1psid or not secure_1psidts:
            logger.error("❌ Could not extract cookies from file")
            return False
        
        logger.info(f"✅ Extracted cookies:")
        logger.info(f"   __Secure-1PSID: {secure_1psid[:30]}...")
        logger.info(f"   __Secure-1PSIDTS: {secure_1psidts[:30]}...")
        
        # Validate them
        is_valid = await validate_gemini_cookies(secure_1psid, secure_1psidts)
        
        return is_valid
        
    except Exception as e:
        logger.error(f"❌ Error validating cookies from file: {e}")
        return False


async def main():
    """Main entry point for cookie validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Gemini cookies')
    parser.add_argument('--cookie-file', type=str, default='cookies.txt', help='Path to cookie file')
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("Gemini Cookie Validation")
    logger.info("=" * 60)
    
    # Validate cookies
    is_valid = await validate_cookies_from_file(args.cookie_file)
    
    logger.info("=" * 60)
    if is_valid:
        logger.success("✅ VALIDATION PASSED - Cookies are valid!")
        logger.info("You can start the server now: python run_modified.py")
        return True
    else:
        logger.error("❌ VALIDATION FAILED - Cookies are invalid or rate limited!")
        logger.info("Solutions:")
        logger.info("  1. Wait 30-60 minutes (if rate limited)")
        logger.info("  2. Run: python auto_cookie_refresh.py --force")
        logger.info("  3. Check your Google account status")
        return False


if __name__ == '__main__':
    result = asyncio.run(main())
    sys.exit(0 if result else 1)













