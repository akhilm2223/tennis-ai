#!/usr/bin/env python
"""
Wrapper script to run backend_server.py from the root directory
"""
import os
import sys

# Change to the tennis-ai-main directory
os.chdir(os.path.join(os.path.dirname(__file__), 'tennis-ai-main'))

# Add the directory to Python path
sys.path.insert(0, os.getcwd())

# Import and run the server
if __name__ == '__main__':
    from backend_server import app, socketio
    import os
    
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("🎾 Tennis Analysis Server Starting...")
    print(f"📡 Server running on http://localhost:{port}")
    print("🔌 WebSocket enabled for real-time updates")
    
    try:
        socketio.run(app, host=host, port=port, debug=True)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {port} is already in use.")
            print(f"💡 Try one of these solutions:")
            print(f"   1. Kill the process using port {port}")
            print(f"   2. Use a different port: PORT=5002 python run_backend.py")
            sys.exit(1)
        else:
            raise

