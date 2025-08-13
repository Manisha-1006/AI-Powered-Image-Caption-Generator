#!/usr/bin/env python3
"""
Run script for AI Image Caption Generator
"""

import os
import sys

def main():
    """Main function to run the image analyzer"""
    print("=" * 60)
    print("🤖 AI-Powered Image Caption Generator")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  WARNING: OpenAI API key not found!")
        print("   Please set the OPENAI_API_KEY environment variable")
        print("   The application will still run but image analysis will not work")
        print()
    else:
        print("✅ OpenAI API key found")
        print()
    
    print("🚀 Starting application...")
    print("📱 Open your browser and go to: http://localhost:7860")
    print("🔄 The server will start in a few seconds...")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Import and run the main application
    try:
        from simple_image_analyzer import create_gradio_interface
        
        app = create_gradio_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
