#!/usr/bin/env python3
"""
CLAMS Agent Prototype Launcher

Launch the CLAMS Agent Prototype application with AG-UI integration.
"""

import os
import sys
import subprocess
import argparse

def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("🤖 CLAMS Agent Prototype")
    print("   Automated Pipeline Generation for Video Analysis")
    print("=" * 60)
    print()

def print_features():
    """Print application features."""
    print("🚀 Features:")
    print("   • AG-UI integration for real-time communication")
    print("   • Streaming responses with live updates")
    print("   • Modern LangGraph patterns")
    print("   • Interactive pipeline visualization")
    print("   • Session-based conversation management")
    print("   • Human-in-the-loop validation")
    print()

def run_app():
    """Run the application."""
    print("🚀 Starting CLAMS Agent Prototype...")
    print("   Features: AG-UI integration, streaming, LangGraph")
    print("   URL: http://localhost:5000")
    print()
    
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        return 1
    except FileNotFoundError:
        print("❌ app.py not found. Make sure you're in the correct directory.")
        return 1
    
    return 0

def interactive_mode():
    """Run in interactive mode."""
    print_banner()
    print_features()
    
    print("Press Enter to start the application, or Ctrl+C to exit...")
    try:
        input()
        return run_app()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CLAMS Agent Prototype Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py              # Interactive mode with banner
  python run.py --direct     # Run directly without banner
        """
    )
    
    parser.add_argument(
        '--direct', 
        action='store_true',
        help='Run directly without interactive banner'
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found.")
        print("Make sure you're in the CLAMS Agent Prototype directory.")
        return 1
    
    if args.direct:
        return run_app()
    else:
        return interactive_mode()

if __name__ == '__main__':
    sys.exit(main())