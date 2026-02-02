
#!/usr/bin/env python3
import sys
from pathlib import Path
import uvicorn

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.main import create_app
from loguru import logger

def main():
    logger.info("🚀 Starting Gemini-FastAPI Server (FAST MODE - No Auth Check)")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=30000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
