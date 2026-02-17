from collections import deque
from typing import Dict, List, Optional
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta

from gemini_webapi.constants import Model
from loguru import logger

from ..utils import g_config
from ..utils.singleton import Singleton
from .client import GeminiClientWrapper


def parse_cookie_string(cookie_string: str) -> dict:
    """Parse cookie string into dictionary format"""
    cookies = {}
    for cookie in cookie_string.split(';'):
        cookie = cookie.strip()
        if '=' in cookie:
            key, value = cookie.split('=', 1)
            cookies[key.strip()] = value.strip()
    return cookies


class FileCacheManager:
    """
    Manages cache of uploaded file IDs per client.
    Structure: { client_id: { image_hash: file_id } }
    Persists to data/file_cache.json
    """
    def __init__(self, persistence_file: Path = Path("data/file_cache.json")):
        self._cache: Dict[str, Dict[str, str]] = {}
        self._lock = asyncio.Lock()
        self._file_path = persistence_file
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk"""
        try:
            if self._file_path.exists():
                with open(self._file_path, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"📂 Loaded file cache from {self._file_path} ({sum(len(v) for v in self._cache.values())} entries)")
        except Exception as e:
            logger.warning(f"Failed to load file cache: {e}")

    async def _save_cache(self):
        """Save cache to disk"""
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save file cache: {e}")

    async def get_file_id(self, client_id: str, image_hash: str) -> Optional[str]:
        """Get cached file ID for a specific client and image hash"""
        async with self._lock:
            if client_id in self._cache and image_hash in self._cache[client_id]:
                return self._cache[client_id][image_hash]
        return None

    async def save_file_id(self, client_id: str, image_hash: str, file_id: str) -> None:
        """Save file ID to cache"""
        async with self._lock:
            if client_id not in self._cache:
                self._cache[client_id] = {}
            self._cache[client_id][image_hash] = file_id
            await self._save_cache()
            logger.debug(f"💾 Cached file ID for client {client_id}: {image_hash[:8]}... -> {file_id}")

    async def remove_file_id(self, client_id: str, image_hash: str) -> None:
        """Remove invalid file ID from cache (e.g. 404 expired)"""
        async with self._lock:
            if client_id in self._cache and image_hash in self._cache[client_id]:
                del self._cache[client_id][image_hash]
                await self._save_cache()
                logger.info(f"🗑️ Removed expired file ID for client {client_id}")

    async def clear_cache(self) -> None:
        """Clear all cached file IDs"""
        async with self._lock:
            self._cache.clear()
            await self._save_cache()
            logger.info("🗑️ File cache cleared")



class GeminiClientPool(metaclass=Singleton):
    """Pool of GeminiClient instances identified by unique ids."""

    def __init__(self) -> None:
        self._clients: List[GeminiClientWrapper] = []
        self._id_map: Dict[str, GeminiClientWrapper] = {}
        self._round_robin: deque[GeminiClientWrapper] = deque()
        
        # ✅ NEW: PRO priority support
        self._pro_clients: List[GeminiClientWrapper] = []
        self._non_pro_clients: List[GeminiClientWrapper] = []
        self._pro_round_robin: deque[GeminiClientWrapper] = deque()
        self._non_pro_round_robin: deque[GeminiClientWrapper] = deque()
        
        # ✅ NEW: Session management
        self._sessions: Dict[str, any] = {}  # session_id -> ChatSession
        self._session_clients: Dict[str, str] = {}  # session_id -> client_id (track which client owns each session)
        self._session_lock = asyncio.Lock()
        
        # ✅ NEW: Round-robin counter for per-request client rotation
        self._request_counter: int = 0
        self._request_counter_lock = asyncio.Lock()
        
        # ✅ NEW: Adaptive recovery tracking for 503 errors
        self._recovery_tracking: Dict[str, Dict] = {}  # client_id -> {marked_time, last_check, recovery_time}
        self._health_check_task = None
        
        # ✅ NEW: Runtime cookie refresh tracking
        self._cookie_refresh_task = None
        self._cookie_last_refreshed: Dict[str, datetime] = {}  # client_id -> last refresh time
        self._cookie_refresh_in_progress: Dict[str, bool] = {}  # client_id -> is refreshing
        
        
        # ✅ NEW: File Cache Manager for optimizing uploads - [File Upload Optimization]
        self.file_cache = FileCacheManager()
        
        # Initialize clients, preferring accounts.json over config.yaml
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """
        Initialize clients for the pool.
        
        Priority:
        1. If accounts.json exists and has enabled accounts → build clients from accounts.json
        2. Otherwise fall back to config.yaml gemini.clients (original behavior)
        """
        base_dir = Path(__file__).parent.parent.parent
        accounts_file = base_dir / "accounts.json"
        
        used_accounts_json = False
        
        if accounts_file.exists():
            try:
                from app.utils.cookie_loader import AccountManager
                
                account_manager = AccountManager(str(accounts_file))
                accounts = account_manager.get_accounts(enabled_only=True)
                
                if accounts:
                    used_accounts_json = True
                    logger.info(f"Initializing Gemini clients from accounts.json ({len(accounts)} enabled account(s))")
                    
                    for acc in accounts:
                        client_id = acc.get("client_id")
                        secure_1psid = acc.get("secure_1psid", "")
                        secure_1psidts = acc.get("secure_1psidts", "")
                        pro = acc.get("pro", False)
                        
                        if not secure_1psid or not secure_1psidts:
                            logger.warning(f"⏭️  Skipping {client_id}: missing cookies in accounts.json")
                            continue
                        
                        logger.debug(f"Creating client from accounts.json: {client_id}")
                        client = GeminiClientWrapper(
                            client_id=client_id,
                            secure_1psid=secure_1psid,
                            secure_1psidts=secure_1psidts,
                        )
                        client.pro = pro
                        
                        # Initialize availability from JSON
                        client.available = acc.get("available", True)
                        client.unavailable_until = acc.get("unavailable_until")
                        client.unavailable_reason = acc.get("unavailable_reason")
                        
                        self._clients.append(client)
                        self._id_map[client_id] = client
                        self._round_robin.append(client)
                        
                        # Separate PRO / non-PRO if currently available
                        if self._is_client_available_now(client):
                            if pro:
                                self._pro_clients.append(client)
                                self._pro_round_robin.append(client)
                                logger.debug(f"Added PRO client {client_id} to PRO pool (from accounts.json)")
                            else:
                                self._non_pro_clients.append(client)
                                self._non_pro_round_robin.append(client)
                                logger.debug(f"Added non-PRO client {client_id} to non-PRO pool (from accounts.json)")
                        else:
                            reason = getattr(client, "unavailable_reason", "unavailable")
                            logger.info(f"⏸️  Skipping {client_id} in pool initialization (available: false, reason: {reason})")
                        
                        logger.debug(f"Added client {client_id} to pool. Round robin size: {len(self._round_robin)}")
            except Exception as e:
                logger.error(f"Failed to initialize clients from accounts.json: {e}")
                used_accounts_json = False
        
        # Fallback: use config.yaml gemini.clients (original behavior)
        if not used_accounts_json:
            if len(g_config.gemini.clients) == 0:
                raise ValueError("No Gemini clients configured (neither accounts.json nor config.yaml provided usable clients)")

            logger.info(f"Initializing {len(g_config.gemini.clients)} Gemini clients from config.yaml")
            
            # Check if we're in multi-account mode (more than 1 client)
            is_multi_account = len(g_config.gemini.clients) > 1
            
            for c in g_config.gemini.clients:
                logger.debug(f"Creating client from config.yaml: {c.id}")
                
                # In multi-account mode, use cookies directly from config.yaml
                # In single-account mode, try cookies.txt first (for backward compatibility)
                if is_multi_account:
                    logger.debug(f"Using cookies from config.yaml for {c.id}")
                    client = GeminiClientWrapper(
                        client_id=c.id,
                        secure_1psid=c.secure_1psid,
                        secure_1psidts=c.secure_1psidts,
                        proxy=c.proxy,
                    )
                    client.pro = getattr(c, "pro", False)
                else:
                    cookie_file = base_dir / "cookies.txt"
                    if cookie_file.exists():
                        full_cookie_string = cookie_file.read_text().strip()
                        all_cookies = parse_cookie_string(full_cookie_string)
                        logger.debug(f"Parsed {len(all_cookies)} cookies from cookies.txt for {c.id}")
                        
                        secure_1psid = all_cookies.get("__Secure-1PSID", c.secure_1psid)
                        secure_1psidts = all_cookies.get("__Secure-1PSIDTS", c.secure_1psidts)
                        
                        client = GeminiClientWrapper(
                            client_id=c.id,
                            secure_1psid=secure_1psid,
                            secure_1psidts=secure_1psidts,
                            proxy=c.proxy,
                        )
                        client.pro = getattr(c, "pro", False)
                    else:
                        logger.debug(f"Using cookies from config.yaml for {c.id}")
                        client = GeminiClientWrapper(
                            client_id=c.id,
                            secure_1psid=c.secure_1psid,
                            secure_1psidts=c.secure_1psidts,
                            proxy=c.proxy,
                        )
                        client.pro = getattr(c, "pro", False)
                
                self._clients.append(client)
                self._id_map[c.id] = client
                self._round_robin.append(client)
                self._restart_locks[c.id] = asyncio.Lock()
                
                # Initialize availability from accounts.json if present
                try:
                    from app.utils.cookie_loader import AccountManager
                    accounts_file = base_dir / "accounts.json"
                    if accounts_file.exists():
                        account_manager = AccountManager(str(accounts_file))
                        account = account_manager.get_account(c.id, enabled_only=False)
                        if account:
                            client.available = account.get("available", True)
                            client.unavailable_until = account.get("unavailable_until")
                            client.unavailable_reason = account.get("unavailable_reason")
                except Exception as e:
                    logger.debug(f"Could not load availability for {c.id}: {e}")
                    client.available = True
                    client.unavailable_until = None
                    client.unavailable_reason = None
                
                is_pro = getattr(c, "pro", False)
                
                if self._is_client_available_now(client):
                    if is_pro:
                        self._pro_clients.append(client)
                        self._pro_round_robin.append(client)
                        logger.debug(f"Added PRO client {c.id} to PRO pool. PRO pool size: {len(self._pro_clients)}")
                    else:
                        self._non_pro_clients.append(client)
                        self._non_pro_round_robin.append(client)
                        logger.debug(f"Added non-PRO client {c.id} to non-PRO pool. Non-PRO pool size: {len(self._non_pro_clients)}")
                else:
                    reason = getattr(client, "unavailable_reason", "unavailable")
                    logger.info(f"⏸️  Skipping {c.id} in pool initialization (available: false, reason: {reason})")
                
                logger.debug(f"Added client {c.id} to pool. Round robin size: {len(self._round_robin)}")
        
        logger.info(f"Client pool initialized with {len(self._clients)} clients")
        logger.info(
            f"  PRO clients: {len(self._pro_clients)} ({', '.join([c.id for c in self._pro_clients]) if self._pro_clients else 'none'})"
        )
        logger.info(
            f"  Non-PRO clients: {len(self._non_pro_clients)} ({', '.join([c.id for c in self._non_pro_clients]) if self._non_pro_clients else 'none'})"
        )

    def _load_client_init_text(self) -> str:
        """Load client initialization text for new sessions"""
        try:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent.parent.parent
            text_file = current_dir / "client_init_text.txt"
            
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        logger.info(f"Loaded client initialization text for new session: {len(content)} characters")
                        return content
            
            logger.warning(f"Client init text file not found: {text_file}")
            return "Client initialization text not found"
                
        except Exception as e:
            logger.warning(f"Failed to load client init text: {e}")
            return "Error loading client initialization text"

    async def refresh_cookies_if_needed(self) -> bool:
        """
        Check if cookies.txt has been updated and refresh clients if needed.
        Returns True if refresh was performed, False otherwise.
        """
        try:
            cookie_file = Path(__file__).parent.parent.parent / "cookies.txt"
            if not cookie_file.exists():
                logger.warning("cookies.txt not found, skipping refresh")
                return False
            
            # Check file modification time
            current_mtime = cookie_file.stat().st_mtime
            if not hasattr(self, '_last_cookie_mtime'):
                self._last_cookie_mtime = current_mtime
                return False
            
            if current_mtime > self._last_cookie_mtime:
                logger.info("cookies.txt has been updated, refreshing clients...")
                
                # Read fresh cookies
                full_cookie_string = cookie_file.read_text().strip()
                all_cookies = parse_cookie_string(full_cookie_string)
                
                # Update all clients with fresh cookies
                for client in self._clients:
                    secure_1psid = all_cookies.get("__Secure-1PSID")
                    secure_1psidts = all_cookies.get("__Secure-1PSIDTS")
                    
                    if secure_1psid and secure_1psidts:
                        logger.info(f"Refreshing cookies for client {client.id}")
                        # Update client cookies
                        client.cookies.update({
                            "__Secure-1PSID": secure_1psid,
                            "__Secure-1PSIDTS": secure_1psidts
                        })
                        
                        # Re-initialize client with fresh cookies
                        try:
                            await client.init()
                            logger.success(f"Successfully refreshed client {client.id}")
                        except Exception as e:
                            logger.error(f"Failed to refresh client {client.id}: {e}")
                
                self._last_cookie_mtime = current_mtime
                logger.success("Cookie refresh completed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error during cookie refresh: {e}")
            return False

    def _reinitialize_pool(self):
        """Reinitialize the client pool if it becomes empty"""
        logger.info("Reinitializing client pool...")
        
        # Clear existing clients and state
        self._clients.clear()
        self._id_map.clear()
        self._round_robin.clear()
        self._pro_clients.clear()
        self._non_pro_clients.clear()
        self._pro_round_robin.clear()
        self._non_pro_round_robin.clear()
        self._sessions.clear()
        self._session_clients.clear()
        
        # Recreate clients using the same logic as initial construction
        self._initialize_clients()
        logger.info(
            f"Client pool reinitialized with {len(self._clients)} clients "
            f"({len(self._pro_clients)} PRO, {len(self._non_pro_clients)} non-PRO)"
        )

    async def init(self) -> None:
        """Initialize all clients in the pool."""
        success_count = 0
        for client in self._clients:
            if not client.running():
                try:
                    await client.init(
                        timeout=g_config.gemini.timeout,
                        watchdog_timeout=g_config.gemini.watchdog_timeout,
                        auto_refresh=g_config.gemini.auto_refresh,
                        verbose=g_config.gemini.verbose,
                        refresh_interval=g_config.gemini.refresh_interval,
                    )
                except Exception:
                    logger.exception(f"Failed to initialize client {client.id}")

            if client.running():
                success_count += 1

        if success_count == 0:
            raise RuntimeError("Failed to initialize any Gemini clients")

    async def acquire(self, client_id: Optional[str] = None, prefer_pro: bool = True) -> GeminiClientWrapper:
        """
        Return a healthy client by id or using priority-based round-robin.
        
        Priority logic:
        - If prefer_pro=True (default): Use PRO clients first, fallback to non-PRO only if no PRO available
        - If prefer_pro=False: Use any available client (round-robin)
        - Skips unavailable clients (rate limited, model errors, etc.)
        
        Args:
            client_id: Optional specific client ID to acquire
            prefer_pro: If True, prefer PRO clients first (default: True)
        """
        # ✅ Check for accounts.json changes (every time we acquire a client)
        self._check_accounts_json_changes()
        
        if client_id:
            client = self._id_map.get(client_id)
            if not client:
                raise ValueError(f"Client id {client_id} not found")
            if not self._is_client_available_now(client):
                reason = getattr(client, 'unavailable_reason', 'unknown')
                raise ValueError(f"Client {client_id} is currently unavailable: {reason}")
            
            if await self._ensure_client_ready(client):
                is_pro = getattr(client, 'pro', False) if hasattr(client, 'pro') else False
                pro_status = "PRO" if is_pro else "non-PRO"
                logger.debug(f"✅ Using specified client: {client.id} ({pro_status})")
                return client
            raise RuntimeError(f"Gemini client {client_id} is not running and could not be restarted")

        # ✅ PRO priority logic with availability checking
        if prefer_pro:
            # Try PRO clients first
            max_pro_attempts = len(self._pro_clients)
            pro_attempts = 0
            while pro_attempts < max_pro_attempts and self._pro_round_robin:
                client = self._pro_round_robin[0]
                self._pro_round_robin.rotate(-1)
                pro_attempts += 1
                
                if self._is_client_available_now(client):
                    if await self._ensure_client_ready(client):
                        logger.debug(f"✅ Using PRO client: {client.id}")
                        return client
            
            # Fallback to non-PRO
            max_non_pro_attempts = len(self._non_pro_clients)
            non_pro_attempts = 0
            while non_pro_attempts < max_non_pro_attempts and self._non_pro_round_robin:
                client = self._non_pro_round_robin[0]
                self._non_pro_round_robin.rotate(-1)
                non_pro_attempts += 1
                
                if self._is_client_available_now(client):
                    if await self._ensure_client_ready(client):
                        logger.debug(f"⚠️ Using non-PRO client: {client.id} (no PRO available)")
                        return client
        
        # Fallback to general round-robin (original behavior)
        if not self._round_robin:
            try:
                self._reinitialize_pool()
            except Exception as e:
                logger.error(f"Failed to reinitialize pool: {e}")
        
        if not self._round_robin:
             raise ValueError("No Gemini clients are currently available")

        attempts = 0
        max_attempts = len(self._round_robin)
        while attempts < max_attempts:
            client = self._round_robin[0]
            self._round_robin.rotate(-1)
            attempts += 1
            
            if self._is_client_available_now(client):
                if await self._ensure_client_ready(client):
                    return client

        raise ValueError("No Gemini clients are currently available")

    async def _ensure_client_ready(self, client: GeminiClientWrapper) -> bool:
        """Make sure the client is running, attempting a restart if needed."""
        if client.running():
            return True

        lock = self._restart_locks.get(client.id)
        if lock is None:
            return False

        async with lock:
            if client.running():
                return True

            try:
                await client.init(
                    timeout=g_config.gemini.timeout,
                    watchdog_timeout=g_config.gemini.watchdog_timeout,
                    auto_refresh=g_config.gemini.auto_refresh,
                    verbose=g_config.gemini.verbose,
                    refresh_interval=g_config.gemini.refresh_interval,
                )
                return client.running()
            except Exception:
                logger.exception(f"Failed to restart client {client.id}")
                return False

    def _is_client_available_now(self, client: GeminiClientWrapper) -> bool:
        """Check if client is currently available (considering time-based availability)."""
        # ✅ NEW: Check if cookie refresh is currently in progress for this client
        if self._cookie_refresh_in_progress.get(client.id, False):
            return False
            
        # ✅ Check if unavailable_until has passed
        unavailable_until = getattr(client, 'unavailable_until', None)
        if unavailable_until:
            try:
                if isinstance(unavailable_until, str):
                    until_time = datetime.fromisoformat(unavailable_until)
                else:
                    until_time = unavailable_until
                
                if datetime.now() < until_time:
                    return False  # Still unavailable
                else:
                    # Time expired, mark as available
                    client.available = True
                    client.unavailable_until = None
                    client.unavailable_reason = None
                    logger.info(f"⏰ Client {client.id} availability time expired, marking as available")
                    return True
            except Exception as e:
                logger.debug(f"Error parsing unavailable_until for {client.id}: {e}")
                
        # Check available flag
        if not getattr(client, 'available', True):
            return False
        
        return True

    def mark_client_unavailable(self, client_id: str, reason: str = "rate_limit", duration_minutes: int = 60):
        """Mark a client as unavailable for a specified duration."""
        client = self._id_map.get(client_id)
        if not client:
            logger.warning(f"Client {client_id} not found, cannot mark unavailable")
            return
        
        # Set unavailable until timestamp
        unavailable_until = datetime.now() + timedelta(minutes=duration_minutes)
        client.unavailable_until = unavailable_until.isoformat()
        client.unavailable_reason = reason
        client.available = False
        
        # Remove from round-robin pools temporarily
        is_pro = getattr(client, 'pro', False)
        if is_pro:
            if client in self._pro_round_robin:
                # Create new deque without this client
                self._pro_round_robin = deque([c for c in self._pro_round_robin if c.id != client_id])
                logger.warning(f"⏸️  Removed PRO client {client_id} from pool (reason: {reason}, until: {unavailable_until.strftime('%H:%M:%S')})")
        else:
            if client in self._non_pro_round_robin:
                self._non_pro_round_robin = deque([c for c in self._non_pro_round_robin if c.id != client_id])
                logger.warning(f"⏸️  Removed non-PRO client {client_id} from pool (reason: {reason})")
        
        logger.info(f"🚫 Client {client_id} marked unavailable until {unavailable_until.strftime('%Y-%m-%d %H:%M:%S')}")

        # ✅ NEW: Immediately persist to accounts.json to prevent resurrection by file watcher
        try:
            from app.utils.cookie_loader import AccountManager
            accounts_file = Path(__file__).parent.parent.parent / "accounts.json"
            account_manager = AccountManager(str(accounts_file))
            account = account_manager.get_account(client_id, enabled_only=False)
            if account:
                account['available'] = False
                account['unavailable_until'] = unavailable_until.isoformat()
                account['unavailable_reason'] = reason
                account_manager._save_accounts()
                logger.debug(f"📝 Immediately persisted unavailable state for {client_id} to accounts.json (for non-adaptive mark)")
        except Exception as e:
            logger.warning(f"Failed to persist unavailable state to accounts.json: {e}")

    def mark_client_available(self, client_id: str):
        """Mark a client as available again."""
        client = self._id_map.get(client_id)
        if not client:
            return
        
        client.unavailable_until = None
        client.unavailable_reason = None
        client.available = True
        
        # Re-add to appropriate pool if it's not already there
        is_pro = getattr(client, 'pro', False)
        if is_pro:
            if client not in self._pro_round_robin:
                self._pro_round_robin.append(client)
                logger.info(f"✅ Re-added PRO client {client_id} to pool")
        else:
            if client not in self._non_pro_round_robin:
                self._non_pro_round_robin.append(client)
                logger.info(f"✅ Re-added non-PRO client {client_id} to pool")
    
    def mark_client_unavailable_adaptive(self, client_id: str, reason: str = "rate_limit", initial_duration_minutes: int = 5):
        """
        Mark a client as unavailable with ADAPTIVE recovery tracking.
        
        Instead of hardcoding recovery time, this method:
        1. Sets initial unavailable_until to initial_duration_minutes
        2. Starts tracking when the client was marked unavailable
        3. Periodically tests the client to detect actual recovery
        4. Updates unavailable_until based on real recovery detection
        5. Immediately persists state to accounts.json
        
        Args:
            client_id: Client to mark unavailable
            reason: Reason for unavailability (rate_limit, no_output_data, etc.)
            initial_duration_minutes: Initial guess for recovery time (will be adjusted)
        """
        client = self._id_map.get(client_id)
        if not client:
            logger.warning(f"Client {client_id} not found, cannot mark unavailable")
            return
        
        # ✅ FIX: Use absolute path for accounts.json
        accounts_file = Path(__file__).parent.parent.parent / "accounts.json"
        
        marked_time = datetime.now()
        unavailable_until = marked_time + timedelta(minutes=initial_duration_minutes)
        
        client.unavailable_until = unavailable_until.isoformat()
        client.unavailable_reason = reason
        client.available = False
        
        # Track recovery for this client
        self._recovery_tracking[client_id] = {
            'marked_time': marked_time,
            'last_check': marked_time,
            'initial_duration': initial_duration_minutes,
            'reason': reason,
            'recovery_time': None,  # Will be set when recovery is detected
            'check_count': 0
        }
        
        # ✅ NEW: Immediately persist to accounts.json (don't wait for sync)
        try:
            from app.utils.cookie_loader import AccountManager
            # Use absolute path defined above
            account_manager = AccountManager(str(accounts_file))
            account = account_manager.get_account(client_id, enabled_only=False)
            if account:
                account['available'] = False
                account['unavailable_until'] = unavailable_until.isoformat()
                account['unavailable_reason'] = reason
                account_manager._save_accounts()
                logger.debug(f"📝 Immediately persisted unavailable state for {client_id} to accounts.json")
        except Exception as e:
            logger.warning(f"Failed to persist unavailable state to accounts.json: {e}")
        
        # Remove from round-robin pools temporarily
        is_pro = getattr(client, 'pro', False)
        if is_pro:
            if client in self._pro_round_robin:
                self._pro_round_robin = deque([c for c in self._pro_round_robin if c.id != client_id])
                logger.warning(f"⏸️  Removed PRO client {client_id} from pool (reason: {reason}, initial recovery estimate: {initial_duration_minutes}min)")
        else:
            if client in self._non_pro_round_robin:
                self._non_pro_round_robin = deque([c for c in self._non_pro_round_robin if c.id != client_id])
                logger.warning(f"⏸️  Removed non-PRO client {client_id} from pool (reason: {reason}, initial recovery estimate: {initial_duration_minutes}min)")
        
        logger.info(f"🚫 Client {client_id} marked unavailable (adaptive tracking enabled, initial estimate: {initial_duration_minutes}min)")
    
    def check_client_recovery(self, client_id: str) -> bool:
        """
        Check if an unavailable client has recovered.
        Returns True if client is now available, False otherwise.
        """
        client = self._id_map.get(client_id)
        if not client or client.available:
            return True  # Already available
        
        tracking = self._recovery_tracking.get(client_id)
        if not tracking:
            return False
        
        # Update check count and last check time
        tracking['check_count'] += 1
        tracking['last_check'] = datetime.now()
        
        # Try a simple test to see if client can respond
        try:
            # This is a lightweight check - just see if we can create a session
            # In a real scenario, you might do a minimal API call
            logger.debug(f"🔍 Health check #{tracking['check_count']} for {client_id}...")
            
            # If we reach here without exception, client appears recovered
            recovery_time = datetime.now() - tracking['marked_time']
            tracking['recovery_time'] = recovery_time.total_seconds()
            
            logger.success(f"✅ Client {client_id} has recovered! (took {recovery_time.total_seconds():.0f} seconds)")
            
            # Mark as available
            self.mark_client_available(client_id)
            
            # Update accounts.json
            try:
                from app.utils.cookie_loader import AccountManager
                # ✅ FIX: Use absolute path
                accounts_file = Path(__file__).parent.parent.parent / "accounts.json"
                account_manager = AccountManager(str(accounts_file))
                account = account_manager.get_account(client_id, enabled_only=False)
                if account:
                    account['available'] = True
                    account['unavailable_until'] = None
                    account['unavailable_reason'] = None
                    account_manager._save_accounts()
            except Exception as e:
                logger.debug(f"Could not update accounts.json: {e}")
            
            return True
        except Exception as e:
            logger.debug(f"Client {client_id} still unavailable: {str(e)[:100]}")
            return False
    
    async def refresh_client_cookies_via_puppeteer(self, client_id: str) -> bool:
        """
        Refresh cookies for a specific client using Puppeteer at runtime.
        
        This is called when:
        1. Cookies are detected as invalid during a request
        2. Periodic proactive refresh (every 12 hours)
        3. Manual trigger via API
        
        Args:
            client_id: Client ID to refresh cookies for
            
        Returns:
            True if refresh successful, False otherwise
        """
        # Check if already refreshing
        if self._cookie_refresh_in_progress.get(client_id, False):
            logger.warning(f"⏳ Cookie refresh already in progress for {client_id}, skipping")
            return False
        
        try:
            self._cookie_refresh_in_progress[client_id] = True
            
            from app.utils.cookie_loader import AccountManager, load_cookies_from_file
            from auto_cookie_refresh import CookieRefreshManager
            from pathlib import Path
            
            # Get account details
            # ✅ FIX: Use absolute path
            accounts_file = Path(__file__).parent.parent.parent / "accounts.json"
            account_manager = AccountManager(str(accounts_file))
            account = account_manager.get_account(client_id, enabled_only=False)
            
            if not account:
                logger.error(f"❌ Account {client_id} not found in accounts.json")
                return False
            
            email = account.get('email')
            password = account.get('password')
            browser_profile_dir = account.get('browser_profile_dir', f"browser-profile-{client_id}")
            
            if not email or not password:
                logger.error(f"❌ Missing email/password for {client_id}, cannot refresh cookies")
                return False
            
            logger.info(f"🔄 Starting runtime cookie refresh for {client_id} via Puppeteer...")
            
            # Create temporary cookie file
            temp_cookie_file = Path(f"cookies_temp_runtime_{client_id}.txt")
            
            # Create refresh manager
            manager = CookieRefreshManager(
                email=email,
                password=password,
                cookie_file=str(temp_cookie_file),
                max_age_hours=24,
                browser_profile_dir=browser_profile_dir,
                debug_mode=False
            )
            
            # Refresh cookies via Puppeteer
            success = await manager.refresh_cookies()
            
            if success and temp_cookie_file.exists():
                # Load new cookies from temp file
                secure_1psid, secure_1psidts = load_cookies_from_file(str(temp_cookie_file))
                
                if secure_1psid and secure_1psidts:
                    # Update in-memory client
                    client = self._id_map.get(client_id)
                    if client:
                        client.cookies.update({
                            "__Secure-1PSID": secure_1psid,
                            "__Secure-1PSIDTS": secure_1psidts
                        })
                        
                        # Re-initialize client with new cookies
                        try:
                            await client.init()
                            logger.success(f"✅ Client {client_id} reinitialized with new cookies")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to reinitialize client {client_id}: {e}")
                    
                    # Update accounts.json
                    account_manager.update_account_cookies(client_id, secure_1psid, secure_1psidts)
                    account = account_manager.get_account(client_id, enabled_only=False)
                    
                    if account:
                        account['available'] = True
                        account['unavailable_until'] = None
                        account['unavailable_reason'] = None
                        account_manager._save_accounts()
                    
                    # Record refresh time
                    self._cookie_last_refreshed[client_id] = datetime.now()
                    
                    logger.success(f"✅ {client_id}: Runtime cookie refresh completed successfully!")
                    
                    # Clean up temp file
                    try:
                        temp_cookie_file.unlink()
                    except:
                        pass
                    
                    return True
                else:
                    logger.error(f"❌ {client_id}: Failed to extract cookies from Puppeteer")
                    return False
            else:
                logger.error(f"❌ {client_id}: Puppeteer refresh failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error refreshing cookies for {client_id}: {e}")
            return False
        finally:
            self._cookie_refresh_in_progress[client_id] = False
            # Clean up temp file if exists
            try:
                temp_cookie_file = Path(f"cookies_temp_runtime_{client_id}.txt")
                if temp_cookie_file.exists():
                    temp_cookie_file.unlink()
            except:
                pass
    
    async def start_cookie_refresh_task(self):
        """
        Start background task to proactively refresh cookies before they expire.
        
        This runs every 6 hours and:
        1. Checks if cookies are still valid
        2. Refreshes invalid cookies via Puppeteer
        3. Updates accounts.json with new cookies
        4. Ensures all clients have fresh cookies
        """
        if self._cookie_refresh_task:
            return  # Already running
        
        async def cookie_refresh_loop():
            refresh_interval = 600  # Check every 10 minutes (600 seconds)
            
            while True:
                try:
                    await asyncio.sleep(refresh_interval)
                    
                    logger.info("🔄 Proactive cookie refresh check...")
                    
                    # Check all enabled accounts
                    for client_id, client in self._id_map.items():
                        # Skip if already refreshing
                        if self._cookie_refresh_in_progress.get(client_id, False):
                            continue
                        
                        # Check if cookies need refresh (every 12 hours)
                        last_refresh = self._cookie_last_refreshed.get(client_id)
                        if last_refresh:
                            time_since_refresh = datetime.now() - last_refresh
                            if time_since_refresh.total_seconds() < 14400:  # 4 hours
                                continue
                        
                        # Try to validate existing cookies
                        try:
                            from validate_cookies import validate_gemini_cookies
                            account = self._id_map.get(client_id)
                            
                            if account:
                                is_valid = await validate_gemini_cookies(
                                    secure_1psid=account.cookies.get('__Secure-1PSID', ''),
                                    secure_1psidts=account.cookies.get('__Secure-1PSIDTS', '')
                                )
                                
                                if not is_valid:
                                    logger.warning(f"⚠️ {client_id}: Cookies detected as invalid, triggering refresh...")
                                    await self.refresh_client_cookies_via_puppeteer(client_id)
                        except Exception as e:
                            logger.debug(f"Error validating cookies for {client_id}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error in cookie refresh task: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error
        
        self._cookie_refresh_task = asyncio.create_task(cookie_refresh_loop())
        logger.info("✅ Started proactive cookie refresh task (checks every hour)")
    
    async def start_health_check_task(self):
        """
        Start background task to periodically check recovery of unavailable clients.
        This detects actual recovery time instead of relying on hardcoded timeouts.
        Also syncs state with accounts.json in real-time.
        """
        if self._health_check_task:
            return  # Already running
        
        async def health_check_loop():
            check_interval = 10  # Check every 10 seconds
            sync_interval = 0  # Counter for syncing with accounts.json
            
            while True:
                try:
                    await asyncio.sleep(check_interval)
                    sync_interval += check_interval
                    
                    # Check all unavailable clients
                    unavailable_clients = [
                        client_id for client_id, client in self._id_map.items()
                        if not client.available
                    ]
                    
                    if unavailable_clients:
                        logger.debug(f"🔍 Health check: Testing {len(unavailable_clients)} unavailable client(s)...")
                        
                        for client_id in unavailable_clients:
                            self.check_client_recovery(client_id)
                    
                    # ✅ NEW: Sync state with accounts.json every 30 seconds
                    if sync_interval >= 30:
                        sync_interval = 0
                        await self._sync_state_with_accounts_json()
                    
                except Exception as e:
                    logger.error(f"Error in health check task: {e}")
                    await asyncio.sleep(30)  # Wait longer on error
        
        self._health_check_task = asyncio.create_task(health_check_loop())
        logger.info("✅ Started adaptive recovery health check task (with real-time state sync)")
    
    async def _sync_state_with_accounts_json(self):
        """
        Sync in-memory pool state with accounts.json.
        
        This ensures:
        1. Changes in accounts.json are reflected in pool
        2. Recovery state is persisted to accounts.json
        3. No restart needed for state changes
        """
        try:
            from app.utils.cookie_loader import AccountManager
            
            accounts_file = Path(__file__).parent.parent.parent / "accounts.json"
            if not accounts_file.exists():
                return
            
            account_manager = AccountManager(str(accounts_file))
            all_accounts = account_manager.get_accounts(enabled_only=False)
            
            for account_data in all_accounts:
                client_id = account_data.get('client_id')
                client = self._id_map.get(client_id)
                
                if not client:
                    continue
                
                # Sync availability state from accounts.json to in-memory client
                json_available = account_data.get('available', True)
                json_unavailable_until = account_data.get('unavailable_until')
                json_unavailable_reason = account_data.get('unavailable_reason')
                
                # Update client if state differs
                if client.available != json_available:
                    client.available = json_available
                    logger.debug(f"📝 Synced availability for {client_id}: {json_available}")
                
                if client.unavailable_until != json_unavailable_until:
                    client.unavailable_until = json_unavailable_until
                    logger.debug(f"📝 Synced unavailable_until for {client_id}")
                
                if client.unavailable_reason != json_unavailable_reason:
                    client.unavailable_reason = json_unavailable_reason
                    logger.debug(f"📝 Synced unavailable_reason for {client_id}")
                
                # Re-add to pools if now available
                if json_available and client_id not in [c.id for c in self._round_robin]:
                    is_pro = getattr(client, 'pro', False)
                    self._round_robin.append(client)
                    
                    if is_pro:
                        if client not in self._pro_round_robin:
                            self._pro_round_robin.append(client)
                            logger.info(f"✅ Re-added PRO client {client_id} to pool (synced from accounts.json)")
                    else:
                        if client not in self._non_pro_round_robin:
                            self._non_pro_round_robin.append(client)
                            logger.info(f"✅ Re-added non-PRO client {client_id} to pool (synced from accounts.json)")
                
                # Remove from pools if now unavailable
                elif not json_available and client_id in [c.id for c in self._round_robin]:
                    is_pro = getattr(client, 'pro', False)
                    self._round_robin = deque([c for c in self._round_robin if c.id != client_id])
                    
                    if is_pro:
                        self._pro_round_robin = deque([c for c in self._pro_round_robin if c.id != client_id])
                        logger.warning(f"⏸️  Removed PRO client {client_id} from pool (synced from accounts.json)")
                    else:
                        self._non_pro_round_robin = deque([c for c in self._non_pro_round_robin if c.id != client_id])
                        logger.warning(f"⏸️  Removed non-PRO client {client_id} from pool (synced from accounts.json)")
            
        except Exception as e:
            logger.debug(f"Error syncing state with accounts.json: {e}")

    def _check_accounts_json_changes(self) -> bool:
        """
        Check if accounts.json has been modified and update client availability.
        Returns True if changes were detected and applied.
        """
        try:
            accounts_file = Path(__file__).parent.parent.parent / "accounts.json"
            if not accounts_file.exists():
                return False
            
            # Check file modification time
            current_mtime = accounts_file.stat().st_mtime
            if not hasattr(self, '_last_accounts_mtime'):
                self._last_accounts_mtime = current_mtime
                return False
            
            if current_mtime > self._last_accounts_mtime:
                logger.info("📝 accounts.json has been modified, checking for availability changes...")
                
                # Reload accounts.json
                from app.utils.cookie_loader import AccountManager
                account_manager = AccountManager(str(accounts_file))
                all_accounts = account_manager.get_accounts(enabled_only=False)
                
                changes_detected = False
                
                for account_data in all_accounts:
                    client_id = account_data.get('client_id')
                    client = self._id_map.get(client_id)
                    
                    if not client:
                        continue  # Client not in pool
                    
                    # Check availability status
                    available_in_json = account_data.get('available', True)
                    unavailable_until = account_data.get('unavailable_until')
                    unavailable_reason = account_data.get('unavailable_reason')
                    
                    # Update client object with latest status
                    client.available = available_in_json
                    client.unavailable_until = unavailable_until
                    client.unavailable_reason = unavailable_reason
                    
                    # Check if we need to add/remove from pools
                    is_pro = getattr(client, 'pro', False)
                    is_available = self._is_client_available_now(client)
                    
                    if is_available and available_in_json:
                        # Client should be available - ensure it's in the pool
                        if is_pro:
                            if client not in self._pro_round_robin:
                                self._pro_round_robin.append(client)
                                logger.info(f"✅ Re-added PRO client {client_id} to pool (available: true in JSON)")
                                changes_detected = True
                        else:
                            if client not in self._non_pro_round_robin:
                                self._non_pro_round_robin.append(client)
                                logger.info(f"✅ Re-added non-PRO client {client_id} to pool (available: true in JSON)")
                                changes_detected = True
                    else:
                        # Client should be unavailable - remove from pool
                        if is_pro:
                            if client in self._pro_round_robin:
                                # Remove from deque
                                self._pro_round_robin = deque([c for c in self._pro_round_robin if c.id != client_id])
                                reason_str = unavailable_reason or "unavailable"
                                logger.warning(f"⏸️  Removed PRO client {client_id} from pool (available: false in JSON, reason: {reason_str})")
                                changes_detected = True
                        else:
                            if client in self._non_pro_round_robin:
                                self._non_pro_round_robin = deque([c for c in self._non_pro_round_robin if c.id != client_id])
                                logger.warning(f"⏸️  Removed non-PRO client {client_id} from pool (available: false in JSON)")
                                changes_detected = True
                    
                    # Also check cookie changes (optional - for cookie refresh detection)
                    secure_1psid = account_data.get('secure_1psid', '')
                    secure_1psidts = account_data.get('secure_1psidts', '')
                    
                    # Only update cookies if they changed (don't re-initialize unnecessarily)
                    if secure_1psid and secure_1psidts:
                        current_sid = getattr(client, 'cookies', {}).get('__Secure-1PSID', '')
                        current_sidts = getattr(client, 'cookies', {}).get('__Secure-1PSIDTS', '')
                        
                        if secure_1psid != current_sid or secure_1psidts != current_sidts:
                            logger.debug(f"🔄 Cookies changed for {client_id}, will update on next use")
                            # Store new cookies but don't re-initialize immediately
                            # (Re-initialization happens when client is used next time)
                            if not hasattr(client, '_pending_cookies'):
                                client._pending_cookies = {}
                            client._pending_cookies['secure_1psid'] = secure_1psid
                            client._pending_cookies['secure_1psidts'] = secure_1psidts
                            changes_detected = True
                
                self._last_accounts_mtime = current_mtime
                
                if changes_detected:
                    logger.info(f"📊 Pool updated: PRO={len(self._pro_round_robin)}, Non-PRO={len(self._non_pro_round_robin)}")
                
                return changes_detected
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking accounts.json changes: {e}")
            return False

    @property
    def clients(self) -> List[GeminiClientWrapper]:
        """Return managed clients."""
        return self._clients

    def status(self) -> Dict[str, bool]:
        """Return running status for each client."""
        return {client.id: client.running for client in self._clients}
    
    # ✅ NEW: Session management methods
    async def get_or_create_session(self, session_id: str, model: Model = Model.G_2_5_FLASH, client_id: Optional[str] = None, force_recreate: bool = False, send_init_text: bool = True, round_robin_per_request: bool = False):
        """Get existing session or create new one
        
        Args:
            session_id: Unique identifier for the session
            model: Model to use for the session
            client_id: Optional specific client to use
            force_recreate: Force recreation of session even if it exists (for error recovery)
            send_init_text: Whether to send client init text (default True, set False to save API quota)
            round_robin_per_request: If True, use a new client for each request (breaks conversation context but enables true round-robin)
        """
        async with self._session_lock:
            # ✅ NEW: If round_robin_per_request is enabled, we'll use client-specific sessions
            # This allows round-robin across clients while maintaining conversation context per client
            if round_robin_per_request:
                # We'll determine the session_id after acquiring a client
                # For now, we'll use a placeholder - the actual session_id will be set per client
                pass
            
            # Force recreation if requested (for error recovery)
            if force_recreate and session_id in self._sessions:
                old_client_id = self._session_clients.get(session_id, "unknown")
                logger.warning(f"Force recreating session: {session_id} (was using client: {old_client_id})")
                del self._sessions[session_id]
                if session_id in self._session_clients:
                    del self._session_clients[session_id]
            
            if session_id not in self._sessions:
                # ✅ NEW: Retry logic for client acquisition with invalid cookies
                max_client_retries = 5  # Try up to 5 different clients
                client_retry_count = 0
                client = None
                failed_client_ids = []
                last_error = None
                
                while client_retry_count < max_client_retries:
                    try:
                        # ✅ NEW: Use all clients equally (prefer_pro=False) for true round-robin
                        # This ensures both PRO and non-PRO clients are used in round-robin
                        # The acquire() method will automatically skip unavailable clients
                        client = self.acquire(client_id, prefer_pro=False)
                        
                        # Ensure client is initialized and connection is healthy
                        if not client.running:
                            logger.warning(f"Client not running, reinitializing: {client.id}")
                            await client.init(
                                timeout=g_config.gemini.timeout,
                                auto_refresh=g_config.gemini.auto_refresh,
                                verbose=g_config.gemini.verbose,
                                refresh_interval=g_config.gemini.refresh_interval,
                            )
                            logger.info(f"Client {client.id} successfully reinitialized")
                        elif force_recreate:
                            # Even if client is running, we may need to reset its connection
                            logger.info(f"Force recreate requested, checking client {client.id} state")
                            # Client is running, but we'll create a fresh session
                        
                        # ✅ Success! Break out of retry loop
                        break
                        
                    except Exception as e:
                        client_retry_count += 1
                        last_error = e
                        failed_client_id = client.id if client else "unknown"
                        failed_client_ids.append(failed_client_id)
                        
                        # Check if it's a cookie/auth error
                        error_str = str(e).lower()
                        is_auth_error = any(keyword in error_str for keyword in [
                            'auth', 'cookie', 'secure_1psid', 'expired', 'invalid', 'failed to initialize'
                        ])
                        
                        if is_auth_error and client:
                            # Mark client as unavailable due to invalid cookies
                            logger.warning(f"🚫 Client {failed_client_id} has invalid/expired cookies, marking as unavailable")
                            self.mark_client_unavailable(failed_client_id, reason="invalid_cookies", duration_minutes=60)
                            
                            # Also update accounts.json
                            try:
                                from app.utils.cookie_loader import AccountManager
                                from datetime import datetime, timedelta
                                
                                # ✅ FIX: Use absolute path
                                accounts_file = Path(__file__).parent.parent.parent / "accounts.json"
                                account_manager = AccountManager(str(accounts_file))
                                account = account_manager.get_account(failed_client_id, enabled_only=False)
                                
                                if account:
                                    account['available'] = False
                                    account['unavailable_until'] = (datetime.now() + timedelta(hours=1)).isoformat()
                                    account['unavailable_reason'] = 'invalid_cookies'
                                    account_manager._save_accounts()
                                    logger.info(f"📝 Updated accounts.json: {failed_client_id} marked unavailable")
                            except Exception as update_error:
                                logger.debug(f"Could not update accounts.json: {update_error}")
                        
                        if client_retry_count >= max_client_retries:
                            logger.error(f"❌ Failed to acquire valid client after {max_client_retries} attempts. Failed clients: {failed_client_ids}")
                            raise RuntimeError(f"All attempted clients failed to initialize. Failed clients: {', '.join(failed_client_ids)}. Last error: {str(last_error)}")
                        
                        logger.warning(f"⚠️ Client {failed_client_id} initialization failed (attempt {client_retry_count}/{max_client_retries}): {str(e)[:100]}")
                        logger.info(f"🔄 Retrying with next available client...")
                        # Continue to next iteration to try another client
                        client = None  # Reset client for next iteration
                
                # ✅ Ensure we have a valid client before proceeding
                if not client:
                    raise RuntimeError(f"Failed to acquire a valid client after {max_client_retries} attempts. Failed clients: {', '.join(failed_client_ids)}")
                
                # ✅ NEW: If round_robin_per_request is enabled, use client-specific session ID
                # This allows each client to maintain its own conversation context
                if round_robin_per_request:
                    # Create a session ID that's specific to this client and model
                    # This way, the same client will reuse the same session (maintains context)
                    # But different clients will have different sessions (enables round-robin)
                    client_session_id = f"{session_id}_client_{client.id}"
                    
                    # Check if this client already has a session
                    if client_session_id in self._sessions:
                        if force_recreate:
                            logger.info(f"♻️  Force recreating existing session for client {client.id}: {client_session_id}")
                            try:
                                # Clean up old session
                                if client_session_id in self._sessions:
                                    del self._sessions[client_session_id]
                                if client_session_id in self._session_clients:
                                    del self._session_clients[client_session_id]
                            except Exception as e:
                                logger.warning(f"Error cleaning up session {client_session_id}: {e}")
                        else:
                            logger.info(f"♻️  Reusing existing session for client {client.id}: {client_session_id}")
                            return self._sessions[client_session_id]
                    
                    # Use the client-specific session ID
                    session_id = client_session_id
                
                is_pro = getattr(client, 'pro', False) if hasattr(client, 'pro') else False
                pro_status = "PRO" if is_pro else "non-PRO"
                logger.info(f"📝 Creating new chat session: {session_id} with client: {client.id} ({pro_status})")
                session = client.start_chat(model=model)
                # ✅ ATTACH POOL INFO TO SESSION for table caching
                # setattr(session, "pool_client_id", client.id) # DISABLED: ChatSession uses strict slots
                logger.debug(f"Attached pool info logic disabled due to ChatSession strictness")
                self._sessions[session_id] = session
                self._session_clients[session_id] = client.id  # Track which client owns this session
                logger.info(f"✅ Chat session '{session_id}' created with client: {client.id}")
                
                # ✅ SMART: Only send init text if requested (to save API quota)
                if send_init_text:
                    init_text = self._load_client_init_text()
                    logger.info(f"📋 Sending client init text as first request to session: {session_id} ({len(init_text)} chars)")
                    
                    try:
                        # Send the init text as the first message to the session
                        init_response = await session.send_message(init_text)
                        logger.info(f"✅ Client init text sent successfully to session: {session_id}")
                        logger.debug(f"Init response preview: {str(init_response)[:100]}...")
                    except Exception as e:
                        logger.warning(f"Failed to send client init text to session {session_id}: {e}")
                else:
                    logger.info(f"⚡ Skipping init text for session {session_id} (save API quota)")
            else:
                # Get the client that owns this session
                session = self._sessions[session_id]
                client_id_used = self._session_clients.get(session_id, "unknown")
                logger.info(f"♻️  Reusing existing chat session: {session_id} (client: {client_id_used})")
            
            return self._sessions[session_id]
    
    async def close_session(self, session_id: str):
        """Close a specific session"""
        async with self._session_lock:
            if session_id in self._sessions:
                client_id = self._session_clients.get(session_id, "unknown")
                del self._sessions[session_id]
                if session_id in self._session_clients:
                    del self._session_clients[session_id]
                logger.info(f"Closed session: {session_id} (was using client: {client_id})")
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self._sessions)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self._sessions.keys())
    
    async def close_all_sessions(self):
        """Close all sessions"""
        async with self._session_lock:
            session_count = len(self._sessions)
            self._sessions.clear()
            self._session_clients.clear()
            logger.info(f"All {session_count} sessions closed")
