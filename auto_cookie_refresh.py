"""
Automatic Cookie Refresh for Gemini-FastAPI
============================================
This script automatically refreshes Gemini cookies when they're older than 24 hours.
It uses Playwright to login to Gemini and extract fresh cookies.

Usage:
    python auto_cookie_refresh.py              # Check and refresh if needed
    python auto_cookie_refresh.py --force      # Force refresh regardless of age
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


class CookieRefreshManager:
    """Manages automatic cookie refresh for Gemini authentication."""
    
    def __init__(
        self,
        email: str,
        password: str,
        cookie_file: str = "cookies.txt",
        login_url: str = "https://gemini.google.com/app",
        max_age_hours: int = 24,
        browser_profile_dir: str = "browser-profile",
        debug_mode: bool = False
    ):
        """
        Initialize the cookie refresh manager.
        
        Args:
            email: Google account email
            password: Google account password
            cookie_file: Path to the cookies.txt file
            login_url: Gemini login URL
            max_age_hours: Maximum age of cookies in hours before refresh
            browser_profile_dir: Directory for persistent browser profile (saves login state)
            debug_mode: If True, keeps browser open for inspection and debugging
        """
        self.email = email
        self.password = password
        self.cookie_file = Path(cookie_file)
        self.login_url = login_url
        self.max_age_hours = max_age_hours
        self.browser_profile_dir = Path(browser_profile_dir)
        self.debug_mode = debug_mode
    
    def needs_refresh(self) -> bool:
        """
        Check if the cookie file needs to be refreshed.
        
        Returns:
            True if refresh is needed, False otherwise
        """
        if not self.cookie_file.exists():
            logger.warning(f"Cookie file not found: {self.cookie_file}")
            return True
        
        # Get the last modification time
        mtime = datetime.fromtimestamp(self.cookie_file.stat().st_mtime)
        age = datetime.now() - mtime
        
        logger.info(f"Cookie file age: {age}")
        logger.info(f"Max age: {timedelta(hours=self.max_age_hours)}")
        
        if age > timedelta(hours=self.max_age_hours):
            logger.info(f"Cookie file is older than {self.max_age_hours} hours, refresh needed")
            return True
        
        logger.info(f"Cookie file is fresh (age: {age}), no refresh needed")
        return False
    
    async def validate_existing_cookies(self) -> bool:
        """
        Validate existing cookies by checking if they're still valid.
        This avoids unnecessary refresh if cookies are already working.
        
        Returns:
            True if cookies are valid, False otherwise
        """
        if not self.cookie_file.exists():
            return False
        
        try:
            from app.utils.cookie_loader import load_cookies_from_file
            from gemini_webapi import GeminiClient
            from gemini_webapi.exceptions import AuthError
            
            secure_1psid, secure_1psidts = load_cookies_from_file(str(self.cookie_file))
            
            if not secure_1psid or not secure_1psidts:
                logger.debug("No valid cookies found in file")
                return False
            
            # Try to initialize a client with existing cookies
            logger.debug("Validating existing cookies...")
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
            logger.info("✅ Existing cookies are still valid - no refresh needed!")
            return True
            
        except AuthError:
            logger.debug("Existing cookies are invalid/expired")
            return False
        except Exception as e:
            logger.debug(f"Error validating cookies: {e}")
            return False
    
    async def refresh_cookies(self) -> bool:
        """
        Refresh cookies by logging into Gemini and extracting session cookies.
        
        Returns:
            True if successful, False otherwise
        """
        # ✅ NEW: Check if existing cookies are still valid first
        if await self.validate_existing_cookies():
            return True
        
        try:
            from playwright.async_api import async_playwright
            
            logger.info("Starting cookie refresh process...")
            logger.info(f"Login URL: {self.login_url}")
            logger.info(f"Email: {self.email}")
            
            async with async_playwright() as p:
                # Launch browser with persistent context to save login state
                # This avoids verification codes on subsequent runs
                logger.info("Launching browser with persistent profile...")
                logger.info(f"Browser profile directory: {self.browser_profile_dir.absolute()}")
                logger.info("💡 This saves your login state and 'Don't ask again' choices!")
                
                # Create browser profile directory if it doesn't exist
                self.browser_profile_dir.mkdir(parents=True, exist_ok=True)
                
                # Use persistent context - saves cookies, localStorage, and session data
                try:
                    context = await p.chromium.launch_persistent_context(
                        user_data_dir=str(self.browser_profile_dir),
                        headless=False,  # Must be False for persistent context with 2FA
                        viewport={'width': 1280, 'height': 720},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-blink-features=AutomationControlled'
                        ],
                        no_viewport=False,
                        ignore_default_args=['--enable-automation']
                    )
                except Exception as e:
                    # Fallback without channel if it fails
                    logger.warning(f"Could not launch with system Chrome: {e}")
                    logger.info("Falling back to Chromium...")
                    context = await p.chromium.launch_persistent_context(
                        user_data_dir=str(self.browser_profile_dir),
                        headless=False,
                        viewport={'width': 1280, 'height': 720},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage'
                        ]
                    )
                
                # Create a new page from persistent context
                page = context.pages[0] if context.pages else await context.new_page()
                
                try:
                    # Go to Google accounts login page first (more reliable!)
                    google_signin_url = 'https://accounts.google.com/signin'
                    
                    logger.info(f"🔍 Checking login status...")
                    logger.info(f"Navigating to Google login: {google_signin_url}")
                    await page.goto(google_signin_url, wait_until='domcontentloaded', timeout=30000)
                    
                    # Wait a bit for the page to load
                    await page.wait_for_timeout(3000)
                    
                    # Check current URL to see if we need to login
                    current_url = page.url
                    logger.info(f"Current URL: {current_url}")
                    
                    # ✅ NEW: Check for reCAPTCHA challenge (multiple URL patterns)
                    # Pattern 1: challenge/recaptcha
                    # Pattern 2: confirmidentifier (verify you are robot page)
                    is_recaptcha_page = 'challenge/recaptcha' in current_url or 'confirmidentifier' in current_url
                    
                    if is_recaptcha_page:
                        logger.warning("")
                        logger.warning("=" * 60)
                        logger.warning("  🤖  reCAPTCHA / ROBOT VERIFICATION DETECTED!")
                        logger.warning("=" * 60)
                        logger.warning("")
                        logger.warning("Please complete the verification in the browser:")
                        logger.warning("  • Click 'I'm not a robot' checkbox")
                        logger.warning("  • Complete any image challenges if needed")
                        logger.warning("  • The browser will wait until you complete it")
                        logger.warning("")
                        logger.warning(f"Current URL: {current_url}")
                        logger.warning("Monitoring for verification completion...")
                        logger.warning("=" * 60)
                        logger.warning("")
                        
                        # Wait for reCAPTCHA/verification to be completed
                        max_wait_time = 300000  # 5 minutes
                        check_interval = 2000  # Check every 2 seconds
                        elapsed_time = 0
                        
                        while elapsed_time < max_wait_time:
                            await page.wait_for_timeout(check_interval)
                            elapsed_time += check_interval
                            
                            current_url = page.url
                            
                            # Check if verification is completed (URL changes away from recaptcha/confirmidentifier)
                            if 'challenge/recaptcha' not in current_url and 'confirmidentifier' not in current_url:
                                logger.success("✅ Verification completed! URL changed away from verification page")
                                logger.info(f"   New URL: {current_url}")
                                break
                            
                            # Check if reCAPTCHA element disappeared
                            try:
                                recaptcha_element = await page.query_selector('.dMNVAe, [class*="recaptcha"], div:has-text("Confirm you\'re not a robot")')
                                if not recaptcha_element:
                                    logger.info("✅ Verification element disappeared - checking URL...")
                                    await page.wait_for_timeout(3000)
                                    final_url = page.url
                                    if 'challenge/recaptcha' not in final_url and 'confirmidentifier' not in final_url:
                                        logger.success("✅ Verification completed!")
                                        break
                            except:
                                pass
                            
                            # Show progress every 30 seconds
                            if elapsed_time % 30000 == 0:
                                remaining = (max_wait_time - elapsed_time) // 1000
                                logger.info(f"⏳ Still waiting for verification completion... ({remaining} seconds remaining)")
                                logger.info(f"   Current URL: {current_url}")
                        
                        if elapsed_time >= max_wait_time:
                            logger.error("❌ Timeout waiting for verification!")
                            return False
                        
                        # Wait a bit more for page to settle
                        await page.wait_for_timeout(2000)
                        current_url = page.url
                        logger.info(f"URL after verification: {current_url}")
                    
                    # Check if we're still on the login page or already authenticated
                    is_login_page = 'signin' in current_url or 'ServiceLogin' in current_url
                    
                    if is_login_page:
                        logger.info("🔐 Login required, starting Google authentication...")
                        logger.info("We're on Google login page - proceeding with credentials")
                        
                        # We're on accounts.google.com - do standard Google login
                        # Enter email
                        logger.info("📧 Entering email...")
                        try:
                            email_input = await page.wait_for_selector('input[type="email"]', timeout=10000)
                            await email_input.click()
                            await page.wait_for_timeout(500)
                            await email_input.fill(self.email)
                            await page.wait_for_timeout(1000)
                            
                            # Click Next button (use ID selector like reference)
                            logger.info("Clicking Next button...")
                            try:
                                next_button = await page.wait_for_selector('#identifierNext', timeout=5000)
                                await next_button.click()
                            except:
                                # Fallback: press Enter
                                await page.keyboard.press('Enter')
                            
                            await page.wait_for_timeout(3000)
                            logger.info("✅ Email step completed")
                        except Exception as e:
                            logger.warning(f"Email step error: {e}")
                        
                        # Wait for password field
                        logger.info("🔑 Waiting for password field...")
                        try:
                            # Wait for password container first (like reference)
                            password_container = await page.wait_for_selector('#password', timeout=15000)
                            await password_container.scroll_into_view_if_needed()
                            await page.click('#password')
                            await page.wait_for_timeout(500)
                            
                            # Then wait for actual password input
                            password_input = await page.wait_for_selector(
                                'input[type="password"]',
                                timeout=15000
                            )
                            
                            # ✅ FIXED: Use JavaScript to set password value directly
                            # This handles passwords with special characters like &, <, >, etc.
                            # that might be interpreted incorrectly by fill() or type()
                            logger.info("Entering password (using direct value setting for special characters)...")
                            
                            # Method 1: Use JavaScript to set value directly (most reliable for special chars)
                            try:
                                await password_input.evaluate(f'(element) => {{ element.value = {repr(self.password)}; }}')
                                # Trigger input event to ensure the page recognizes the change
                                await password_input.evaluate('(element) => { element.dispatchEvent(new Event("input", { bubbles: true })); }')
                                await password_input.evaluate('(element) => { element.dispatchEvent(new Event("change", { bubbles: true })); }')
                                logger.debug("✅ Password set using JavaScript (handles special characters)")
                            except Exception as js_error:
                                # Fallback: Use fill() method
                                logger.warning(f"JavaScript method failed: {js_error}, using fill() method")
                            await password_input.fill(self.password)
                            
                            await page.wait_for_timeout(1000)
                            
                            # Verify password was entered (optional check)
                            try:
                                entered_value = await password_input.input_value()
                                if entered_value == self.password:
                                    logger.debug("✅ Password entered correctly")
                                else:
                                    logger.warning(f"⚠️ Password verification failed. Expected length: {len(self.password)}, Got length: {len(entered_value)}")
                            except:
                                pass  # Verification failed, but continue anyway
                            
                            # Click Next/Sign in button (use ID selector like reference)
                            logger.info("Clicking Sign in button...")
                            try:
                                signin_button = await page.wait_for_selector('#passwordNext', timeout=5000)
                                await signin_button.click()
                            except:
                                # Fallback: press Enter
                                await page.keyboard.press('Enter')
                            
                            # Wait for navigation to complete
                            logger.info("⏳ Waiting for login to complete...")
                            await page.wait_for_timeout(5000)
                            
                            # ✅ NEW: Check for reCAPTCHA after password entry (multiple URL patterns)
                            current_url = page.url
                            is_recaptcha_page = 'challenge/recaptcha' in current_url or 'confirmidentifier' in current_url
                            
                            if is_recaptcha_page:
                                logger.warning("")
                                logger.warning("=" * 60)
                                logger.warning("  🤖  reCAPTCHA CHALLENGE DETECTED!")
                                logger.warning("=" * 60)
                                logger.warning("")
                                logger.warning("Please complete the reCAPTCHA in the browser:")
                                logger.warning("  • Click 'I'm not a robot' checkbox")
                                logger.warning("  • Complete any image challenges if needed")
                                logger.warning("  • The browser will wait until you complete it")
                                logger.warning("")
                                logger.warning("Monitoring for reCAPTCHA completion...")
                                logger.warning("=" * 60)
                                logger.warning("")
                                logger.warning(f"Current URL: {current_url}")
                                logger.warning("Monitoring for verification completion...")
                                logger.warning("=" * 60)
                                logger.warning("")
                                
                                # Wait for reCAPTCHA/verification to be completed
                                max_wait_time = 300000  # 5 minutes
                                check_interval = 2000  # Check every 2 seconds
                                elapsed_time = 0
                                
                                while elapsed_time < max_wait_time:
                                    await page.wait_for_timeout(check_interval)
                                    elapsed_time += check_interval
                                    
                                    current_url = page.url
                                    
                                    # Check if verification is completed (URL changes away from recaptcha/confirmidentifier)
                                    if 'challenge/recaptcha' not in current_url and 'confirmidentifier' not in current_url:
                                        logger.success("✅ Verification completed! URL changed away from verification page")
                                        logger.info(f"   New URL: {current_url}")
                                        break
                                    
                                    # Check if reCAPTCHA element disappeared
                                    try:
                                        recaptcha_div = await page.query_selector('.dMNVAe, div:has-text("Confirm you\'re not a robot")')
                                        if not recaptcha_div:
                                            logger.info("✅ Verification element disappeared - checking URL...")
                                            await page.wait_for_timeout(3000)
                                            final_url = page.url
                                            if 'challenge/recaptcha' not in final_url and 'confirmidentifier' not in final_url:
                                                logger.success("✅ Verification completed!")
                                                break
                                    except:
                                        pass
                                    
                                    # Show progress every 30 seconds
                                    if elapsed_time % 30000 == 0:
                                        remaining = (max_wait_time - elapsed_time) // 1000
                                        logger.info(f"⏳ Still waiting for verification completion... ({remaining} seconds remaining)")
                                        logger.info(f"   Current URL: {current_url}")
                                
                                if elapsed_time >= max_wait_time:
                                    logger.error("❌ Timeout waiting for verification!")
                                    return False
                                
                                # Wait a bit more for page to settle
                                await page.wait_for_timeout(2000)
                                current_url = page.url
                                logger.info(f"URL after verification: {current_url}")
                            
                            logger.info("✅ Password step completed")
                        except Exception as e:
                            logger.warning(f"Password step error: {e}")
                        
                        # Check for 2FA/Verification
                        logger.info("Checking for 2FA or verification...")
                        try:
                            # ✅ IMPROVED: Check for multiple 2FA indicators
                            # 1. Verification code input
                            # 2. "To help keep your account safe" message (2-step verification)
                            # 3. Challenge page indicators
                            
                            verification_element = None
                            safety_message_found = False
                            
                            # Check for "To help keep your account safe" message first
                            try:
                                # Check page content for the safety message
                                page_text = await page.content()
                                if 'To help keep your account safe' in page_text or 'make sure it\'s really you' in page_text:
                                    safety_message_found = True
                                    logger.warning("")
                                    logger.warning("=" * 60)
                                    logger.warning("  ⚠️  2-STEP VERIFICATION REQUIRED!")
                                    logger.warning("=" * 60)
                                    logger.warning("")
                                    logger.warning("Please complete the 2-step verification:")
                                    logger.warning("  • Enter the verification code from your phone/email")
                                    logger.warning("  • OR approve the login on your phone")
                                    logger.warning("  • Check 'Don't ask again' to skip this next time!")
                                    logger.warning("")
                                    logger.warning("The browser will wait until you complete it.")
                                    logger.warning("=" * 60)
                                    logger.warning("")
                            except:
                                pass
                            
                            # Try to find verification code input
                            try:
                                verification_element = await page.wait_for_selector(
                                    'input[type="tel"], input[aria-label*="code"], [data-challengetype], input[name*="code"]',
                                    timeout=5000
                                )
                            except:
                                # If no input found but safety message exists, still treat as 2FA
                                if safety_message_found:
                                    verification_element = True  # Use as flag
                                else:
                                    verification_element = None
                            
                            if verification_element or safety_message_found:
                                logger.warning("")
                                logger.warning("=" * 60)
                                logger.warning("  ⚠️  VERIFICATION CODE REQUIRED!")
                                logger.warning("=" * 60)
                                logger.warning("")
                                logger.warning("Please complete the verification in the browser:")
                                logger.warning("  • Enter your verification code from email/SMS")
                                logger.warning("  • OR approve the login on your phone")
                                logger.warning("  • Check 'Don't ask again' to skip this next time!")
                                logger.warning("")
                                logger.warning("The browser will stay open until you complete it.")
                                logger.warning("=" * 60)
                                logger.warning("")
                                
                                # ✅ IMPROVED: Multi-method detection for 2FA completion
                                # Uses multiple methods to detect when user completes 2FA:
                                # 1. URL checking (myaccount.google.com, gemini.google.com)
                                # 2. Cookie checking (check if __Secure-1PSID appears)
                                # 3. Page element checking (check for logged-in indicators)
                                # 4. Navigation events (wait for navigation away from challenge page)
                                
                                logger.info("⏳ Waiting for you to complete 2FA verification...")
                                logger.info("   Using multiple detection methods:")
                                logger.info("   • URL monitoring")
                                logger.info("   • Cookie detection")
                                logger.info("   • Page element checking")
                                
                                max_wait_time = 300000  # 5 minutes
                                check_interval = 2000  # Check every 2 seconds
                                elapsed_time = 0
                                last_url = page.url
                                cookies_checked = False
                                
                                while elapsed_time < max_wait_time:
                                    await page.wait_for_timeout(check_interval)
                                    elapsed_time += check_interval
                                    
                                    current_url = page.url
                                    
                                    # ✅ NEW: Check for reCAPTCHA/verification during 2FA wait
                                    is_recaptcha_page = 'challenge/recaptcha' in current_url or 'confirmidentifier' in current_url
                                    was_recaptcha_page = 'challenge/recaptcha' in str(last_url) or 'confirmidentifier' in str(last_url)
                                    
                                    if is_recaptcha_page:
                                        if not was_recaptcha_page:
                                            logger.warning("")
                                            logger.warning("  🤖  reCAPTCHA/Verification detected during 2FA wait!")
                                            logger.warning("  Please complete the verification...")
                                            logger.warning("")
                                    
                                    # Method 1: URL-based detection (fastest)
                                    if current_url != last_url:
                                        logger.info(f"📍 URL changed: {current_url}")
                                        
                                        # If URL changed away from recaptcha/confirmidentifier, that's progress
                                        if not is_recaptcha_page and was_recaptcha_page:
                                            logger.info("✅ Left verification page - continuing to wait for 2FA...")
                                        
                                        last_url = current_url
                                    
                                    # Check for success URLs
                                    if 'myaccount.google.com' in current_url:
                                        logger.success("✅ 2FA completed! Detected myaccount.google.com (Method: URL)")
                                        break
                                    
                                    if 'gemini.google.com' in current_url:
                                        logger.success("✅ 2FA completed! Already on Gemini page (Method: URL)")
                                        break
                                    
                                    # Method 2: Cookie-based detection (most reliable)
                                    # Check if authentication cookies appear (indicates successful login)
                                    if not cookies_checked or elapsed_time % 10000 == 0:  # Check cookies every 10 seconds
                                        try:
                                            all_cookies = await context.cookies()
                                            cookie_names = [c['name'] for c in all_cookies]
                                            
                                            # Check for authentication cookies that indicate successful login
                                            has_secure_1psid = '__Secure-1PSID' in cookie_names
                                            has_auth_cookies = any(name in cookie_names for name in ['SID', 'HSID', 'SSID', 'APISID'])
                                            
                                            if has_secure_1psid:
                                                logger.success("✅ 2FA completed! Detected __Secure-1PSID cookie (Method: Cookie)")
                                                break
                                            
                                            if has_auth_cookies and 'challenge' not in current_url:
                                                logger.info("✅ Detected authentication cookies - login likely successful")
                                                logger.info("   Waiting a bit more to confirm...")
                                                await page.wait_for_timeout(3000)
                                                
                                                # Double-check URL
                                                final_url = page.url
                                                if 'myaccount.google.com' in final_url or 'gemini.google.com' in final_url:
                                                    logger.success("✅ 2FA completed! Confirmed by cookies + URL (Method: Cookie+URL)")
                                                    break
                                            
                                            cookies_checked = True
                                        except Exception as e:
                                            logger.debug(f"Error checking cookies: {e}")
                                    
                                    # Method 3: Page element detection
                                    # Check if verification element disappeared (user completed 2FA)
                                    try:
                                        verification_still_present = await page.query_selector(
                                            'input[type="tel"], input[aria-label*="code"], [data-challengetype]'
                                        )
                                        
                                        if not verification_still_present:
                                            # Verification element disappeared - might be done
                                            logger.info("✅ Verification element disappeared - checking if login succeeded...")
                                            await page.wait_for_timeout(3000)
                                            
                                            final_url = page.url
                                            if 'myaccount.google.com' in final_url or 'gemini.google.com' in final_url:
                                                logger.success("✅ 2FA completed! Verification element gone + URL confirmed (Method: Element+URL)")
                                                break
                                            elif 'challenge' not in final_url and 'signin' not in final_url:
                                                # Not on challenge page anymore
                                                logger.info(f"✅ Left challenge page. Current URL: {final_url}")
                                                logger.info("   Will proceed to check cookies...")
                                                break
                                    except:
                                        pass  # Element check failed, continue with other methods
                                    
                                    # Method 4: Check if we're no longer on challenge/signin page
                                    if 'challenge' not in current_url and 'signin' not in current_url and 'identifier' not in current_url:
                                        # Not on login/challenge page anymore - might be logged in
                                        logger.info(f"✅ Left login/challenge page. Current URL: {current_url}")
                                        logger.info("⏳ Waiting a bit more to confirm login...")
                                        await page.wait_for_timeout(5000)  # Wait 5 seconds
                                        
                                        # Check final URL
                                        final_url = page.url
                                        logger.info(f"Final URL after wait: {final_url}")
                                        
                                        if 'myaccount.google.com' in final_url or 'gemini.google.com' in final_url:
                                            logger.success("✅ 2FA completed! URL navigation confirmed (Method: Navigation)")
                                            break
                                        elif 'challenge' in final_url or 'signin' in final_url:
                                            # Still on login page, continue waiting
                                            logger.info("Still on login page, continuing to wait...")
                                        else:
                                            # Unknown page, might be logged in - check cookies
                                            logger.info("Unknown page after 2FA, checking cookies...")
                                            try:
                                                all_cookies = await context.cookies()
                                                cookie_names = [c['name'] for c in all_cookies]
                                                if '__Secure-1PSID' in cookie_names:
                                                    logger.success("✅ 2FA completed! Cookies confirmed (Method: Navigation+Cookie)")
                                                    break
                                            except:
                                                pass
                                    
                                    # Show progress every 30 seconds
                                    if elapsed_time % 30000 == 0:
                                        remaining = (max_wait_time - elapsed_time) // 1000
                                        logger.info(f"⏳ Still waiting for 2FA completion... ({remaining} seconds remaining)")
                                        logger.info(f"   Current URL: {current_url}")
                                        logger.info(f"   Detection methods active: URL, Cookie, Element, Navigation")
                                
                                if elapsed_time >= max_wait_time:
                                    logger.error("❌ Timeout waiting for 2FA verification!")
                                    logger.error("Please complete the verification and try again.")
                                    return False
                                
                                # Final verification - check URL and cookies
                                final_check_url = page.url
                                logger.info(f"🔍 Final verification after 2FA...")
                                logger.info(f"   Final URL: {final_check_url}")
                                
                                # Check cookies one more time
                                try:
                                    all_cookies = await context.cookies()
                                    cookie_names = [c['name'] for c in all_cookies]
                                    has_secure_1psid = '__Secure-1PSID' in cookie_names
                                    
                                    if has_secure_1psid:
                                        logger.success("✅ 2FA verification confirmed! __Secure-1PSID cookie found")
                                    elif 'myaccount.google.com' in final_check_url or 'gemini.google.com' in final_check_url:
                                        logger.success("✅ 2FA verification confirmed! Success URL detected")
                                    else:
                                        logger.warning(f"⚠️ 2FA may not be complete. URL: {final_check_url}")
                                        logger.info("Will proceed to Gemini and check cookies...")
                                except Exception as e:
                                    logger.warning(f"Could not verify cookies: {e}")
                                    if 'myaccount.google.com' in final_check_url or 'gemini.google.com' in final_check_url:
                                        logger.success("✅ 2FA verification confirmed! Success URL detected")
                            
                        except Exception as e:
                            # No 2FA required or already completed
                            logger.info("No verification required or already handled")
                        
                        # After Google login, we're authenticated!
                        logger.success("✅ Google authentication completed!")
                    else:
                        logger.success("✅ Already authenticated! (Thanks to persistent browser profile)")
                        logger.info("Skipped login - browser profile saved your session!")
                        logger.info("No credentials needed this time 🎉")
                    
                    # ✅ IMPROVED: Navigate to Gemini to get cookies from the actual app
                    # Check current URL first - might already be on Gemini after 2FA
                    current_url = page.url
                    logger.info(f"Current URL after login: {current_url}")
                    
                    if 'gemini.google.com' in current_url:
                        logger.success("✅ Already on Gemini page after login!")
                        logger.info("⏳ Waiting for Gemini to fully load...")
                        await page.wait_for_timeout(5000)  # Wait for page to settle
                    else:
                        # Navigate to Gemini
                        logger.info(f"🎯 Navigating to Gemini app...")
                        try:
                            # Use 'load' which waits for page load event (more reliable than networkidle)
                            await page.goto(self.login_url, wait_until='load', timeout=30000)
                            logger.info("✅ Page load event fired")
                            
                            # Wait a bit more for Gemini to initialize
                            logger.info("⏳ Waiting for Gemini to fully load...")
                            await page.wait_for_timeout(8000)  # Give it 8 seconds to settle
                            
                            logger.success("✅ Gemini page loaded!")
                        except Exception as nav_error:
                            logger.warning(f"⚠️ Navigation timeout: {nav_error}")
                            logger.info("Trying to extract cookies anyway...")
                            await page.wait_for_timeout(3000)
                    
                    # Check final URL
                    current_url = page.url
                    logger.info(f"Final URL: {current_url}")
                    
                    if 'gemini.google.com' in current_url:
                        logger.success("✅ Successfully on Gemini page!")
                    else:
                        logger.warning(f"⚠️ Not on Gemini page: {current_url}")
                        logger.info("Will try to extract cookies anyway...")
                    
                    # Extract ONLY the 2 important cookies directly from browser
                    logger.info("🍪 Extracting cookies from browser...")
                    all_cookies = await context.cookies()
                    
                    if not all_cookies:
                        logger.error("❌ No cookies found in browser!")
                        logger.info("🔒 Closing browser (no cookies)...")
                        
                        # Close page first
                        try:
                            await page.close()
                        except:
                            pass
                        
                        # Close context
                        try:
                            await context.close()
                        except:
                            pass
                        
                        logger.info("✅ Browser closed!")
                        return False
                    
                    logger.info(f"Found {len(all_cookies)} total cookies in browser")
                    
                    # Extract ONLY the 2 important cookies
                    logger.info("🔍 Looking for the 2 required cookies...")
                    secure_1psid = None
                    secure_1psidts = None
                    
                    for cookie in all_cookies:
                        if cookie['name'] == '__Secure-1PSID':
                            secure_1psid = cookie['value']
                            logger.info(f"✅ Found __Secure-1PSID: {secure_1psid[:20]}...")
                        elif cookie['name'] == '__Secure-1PSIDTS':
                            secure_1psidts = cookie['value']
                            logger.info(f"✅ Found __Secure-1PSIDTS: {secure_1psidts[:20]}...")
                    
                    # Verify BOTH cookies are present
                    if not secure_1psid:
                        logger.error("❌ Missing required cookie: __Secure-1PSID")
                        logger.info("Available cookies: " + ', '.join([c['name'] for c in all_cookies]))
                        
                        # Try to wait and retry
                        logger.info("⏳ Waiting 10 more seconds and retrying...")
                        await page.wait_for_timeout(10000)
                        all_cookies = await context.cookies()
                        
                        for cookie in all_cookies:
                            if cookie['name'] == '__Secure-1PSID':
                                secure_1psid = cookie['value']
                                logger.info(f"✅ Found __Secure-1PSID on retry: {secure_1psid[:20]}...")
                                break
                    
                    if not secure_1psidts:
                        logger.error("❌ Missing required cookie: __Secure-1PSIDTS")
                        logger.info("Available cookies: " + ', '.join([c['name'] for c in all_cookies]))
                        
                        # Try to wait and retry
                        logger.info("⏳ Waiting 10 more seconds and retrying...")
                        await page.wait_for_timeout(10000)
                        all_cookies = await context.cookies()
                        
                        for cookie in all_cookies:
                            if cookie['name'] == '__Secure-1PSIDTS':
                                secure_1psidts = cookie['value']
                                logger.info(f"✅ Found __Secure-1PSIDTS on retry: {secure_1psidts[:20]}...")
                                break
                    
                    # Final check: Both must be present
                    if not secure_1psid or not secure_1psidts:
                        logger.error("❌ Failed to get both required cookies!")
                        if not secure_1psid:
                            logger.error("   Missing: __Secure-1PSID")
                        if not secure_1psidts:
                            logger.error("   Missing: __Secure-1PSIDTS")
                        
                        logger.info("🔒 Closing browser (missing cookies)...")
                        
                        # Close page first
                        try:
                            await page.close()
                        except:
                            pass
                        
                        # Close context
                        try:
                            await context.close()
                        except:
                            pass
                        
                        logger.info("✅ Browser closed!")
                        return False
                    
                    # Success! We have both cookies
                    logger.success(f"✅ Successfully extracted both required cookies!")
                    logger.info(f"   __Secure-1PSID length: {len(secure_1psid)} chars")
                    logger.info(f"   __Secure-1PSIDTS length: {len(secure_1psidts)} chars")
                    
                    # ✅ NEW: Validate cookies before saving
                    logger.info("🔍 Validating extracted cookies...")
                    try:
                        from gemini_webapi import GeminiClient
                        from gemini_webapi.exceptions import AuthError
                        
                        test_client = GeminiClient(
                            secure_1psid=secure_1psid,
                            secure_1psidts=secure_1psidts
                        )
                        
                        await test_client.init(
                            timeout=15,
                            auto_close=False,
                            auto_refresh=False,
                            verbose=False
                        )
                        
                        await test_client.close()
                        logger.success("✅ Cookie validation passed! Cookies are valid and working!")
                        
                    except AuthError as e:
                        logger.error(f"❌ Cookie validation failed: {e}")
                        logger.error("Extracted cookies are invalid - will not save them")
                        logger.info("🔒 Closing browser (invalid cookies)...")
                        
                        try:
                            await page.close()
                            await context.close()
                        except:
                            pass
                        
                        return False
                    except Exception as e:
                        logger.warning(f"⚠️ Cookie validation error (but cookies might still work): {e}")
                        logger.info("Proceeding to save cookies anyway...")
                    
                    # Create cookie string with ONLY the 2 important cookies
                    cookie_string = f"__Secure-1PSID={secure_1psid}; __Secure-1PSIDTS={secure_1psidts}"
                    
                    # Save ONLY the 2 important cookies to file
                    logger.info(f"💾 Saving cookies to {self.cookie_file}...")
                    with open(self.cookie_file, 'w', encoding='utf-8') as f:
                        f.write(cookie_string)
                    
                    logger.success(f"✅ Successfully saved 2 required cookies to {self.cookie_file}")
                    logger.info(f"   Cookie file size: {self.cookie_file.stat().st_size} bytes")
                    logger.info(f"   Saved: __Secure-1PSID and __Secure-1PSIDTS")
                    
                    # Save ALL cookies to full cookie file (for local use)
                    full_cookie_file = self.cookie_file.parent / 'cookies_full.txt'
                    logger.info(f"💾 Saving ALL {len(all_cookies)} cookies to {full_cookie_file}...")
                    
                    # Create full cookie string with all cookies
                    full_cookie_parts = []
                    for cookie in all_cookies:
                        cookie_name = cookie.get('name', '')
                        cookie_value = cookie.get('value', '')
                        full_cookie_parts.append(f"{cookie_name}={cookie_value}")
                    
                    full_cookie_string = "; ".join(full_cookie_parts)
                    
                    with open(full_cookie_file, 'w', encoding='utf-8') as f:
                        f.write(full_cookie_string)
                    
                    logger.success(f"✅ Successfully saved ALL cookies to {full_cookie_file}")
                    logger.info(f"   Full cookie file size: {full_cookie_file.stat().st_size} bytes")
                    logger.info(f"   Saved {len(all_cookies)} cookies for full cookie support")
                    
                    # Update file modification time
                    os.utime(self.cookie_file, None)
                    os.utime(full_cookie_file, None)
                    
                    # Debug mode: Keep browser open for inspection
                    if self.debug_mode:
                        logger.warning("")
                        logger.warning("=" * 60)
                        logger.warning("  🐛 DEBUG MODE: Browser will stay open")
                        logger.warning("=" * 60)
                        logger.warning("")
                        logger.warning("The browser will stay open so you can:")
                        logger.warning("  • Inspect elements (F12 / Right-click → Inspect)")
                        logger.warning("  • Test clicking UI elements")
                        logger.warning("  • Check what the login flow actually needs")
                        logger.warning("  • See if login succeeded")
                        logger.warning("")
                        logger.warning("Press ENTER when you're done inspecting...")
                        logger.warning("=" * 60)
                        logger.warning("")
                        
                        # Wait for user input
                        input()
                        
                        logger.info("🔒 Closing browser...")
                        try:
                            await page.close()
                        except:
                            pass
                        try:
                            await context.close()
                        except:
                            pass
                        logger.success("✅ Browser closed!")
                    else:
                        logger.info("🔒 Closing browser...")
                        try:
                            await page.close()
                        except:
                            pass
                        try:
                            await context.close()
                        except:
                            pass
                        logger.success("✅ Browser closed!")
                    
                    return True
                    
                except Exception as e:
                    logger.exception(f"Error during browser automation: {e}")
                    
                    # Take a screenshot for debugging
                    try:
                        screenshot_path = f"login_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        await page.screenshot(path=screenshot_path)
                        logger.info(f"Screenshot saved to {screenshot_path}")
                    except:
                        pass
                    
                    logger.info("🔒 Closing browser (after error)...")
                    
                    # Close page first
                    try:
                        await page.close()
                    except:
                        pass
                    
                    # Close context
                    try:
                        await context.close()
                    except:
                        pass
                    
                    logger.info("✅ Browser closed!")
                    return False
                    
                finally:
                    # Ensure context is closed
                    try:
                        await context.close()
                        logger.info("✅ Context cleanup complete")
                    except:
                        pass
                    
        except ImportError:
            logger.error("Playwright not installed! Install it with: pip install playwright && playwright install chromium")
            return False
        except Exception as e:
            logger.exception(f"Error refreshing cookies: {e}")
            return False
    
    async def check_and_refresh(self, force: bool = False) -> bool:
        """
        Check if cookies need refresh and refresh if needed.
        
        Args:
            force: Force refresh regardless of cookie age or validity
            
        Returns:
            True if cookies are fresh/valid or successfully refreshed, False otherwise
        """
        if force:
            logger.info("Force refresh requested")
            return await self.refresh_cookies()
        
        # ✅ NEW: First check if existing cookies are still valid
        if await self.validate_existing_cookies():
            logger.info("✅ Existing cookies are valid - no refresh needed")
            return True
        
        # Cookies are invalid or expired - check if refresh is needed by age
        if self.needs_refresh():
            logger.info("Cookie refresh needed (invalid or expired)")
            return await self.refresh_cookies()
        else:
            # Cookies are fresh by age but invalid - refresh anyway
            logger.warning("Cookies are fresh by age but invalid - refreshing anyway")
            return await self.refresh_cookies()


async def main():
    """Main entry point for the cookie refresh script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatically refresh Gemini cookies')
    parser.add_argument('--force', action='store_true', help='Force refresh regardless of cookie age')
    parser.add_argument('--email', type=str, help='Google account email (overrides config)')
    parser.add_argument('--password', type=str, help='Google account password (overrides config)')
    parser.add_argument('--cookie-file', type=str, default='cookies.txt', help='Path to cookie file')
    parser.add_argument('--max-age', type=int, default=24, help='Maximum cookie age in hours')
    parser.add_argument('--debug', action='store_true', help='Debug mode: keep browser open for inspection')
    parser.add_argument('--browser-profile', type=str, default='browser-profile', help='Browser profile directory')
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # Get credentials from args or environment
    email = args.email or os.getenv('GEMINI_EMAIL', 'trungthuphan160@gmail.com')
    password = args.password or os.getenv('GEMINI_PASSWORD', '123456.*@a')
    
    if not email or not password:
        logger.error("Email and password are required!")
        logger.info("Set GEMINI_EMAIL and GEMINI_PASSWORD environment variables or use --email and --password")
        return False
    
    # Create manager
    manager = CookieRefreshManager(
        email=email,
        password=password,
        cookie_file=args.cookie_file,
        max_age_hours=args.max_age,
        browser_profile_dir=args.browser_profile,
        debug_mode=args.debug
    )
    
    # Check and refresh
    logger.info("=" * 60)
    logger.info("Gemini Cookie Auto-Refresh")
    logger.info("=" * 60)
    
    success = await manager.check_and_refresh(force=args.force)
    
    if success:
        logger.success("✅ Cookie refresh completed successfully!")
        return True
    else:
        logger.error("❌ Cookie refresh failed!")
        return False


if __name__ == '__main__':
    result = asyncio.run(main())
    sys.exit(0 if result else 1)









