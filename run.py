import sys
import os

# Ensure the root directory is on the python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.api.app import create_app

if __name__ == "__main__":
    app = create_app("config.yaml")
    
    # Retrieve configuration for running the server
    host = app.config.get('app', {}).get('host', '127.0.0.1')
    port = app.config.get('app', {}).get('port', 5000)
    debug = app.config.get('app', {}).get('debug', True)
    
    app.run(host=host, port=port, debug=debug)
