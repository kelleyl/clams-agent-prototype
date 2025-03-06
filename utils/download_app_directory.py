import requests
import json
import os
from typing import Dict, Any, Optional
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_app_metadata(app_name: str, version: str) -> Dict[str, Any]:
    """
    Fetches the metadata.json for a specific app version
    
    Args:
        app_name: Name of the CLAMS app
        version: Version string (e.g., 'v7.5')
        
    Returns:
        Dictionary containing the app metadata or empty dict if fetch fails
    """
    base_url = "https://apps.clams.ai/"
    metadata_url = urljoin(base_url, f"{app_name}/{version}/metadata.json")
    
    try:
        logger.debug(f"Fetching metadata from {metadata_url}")
        response = requests.get(metadata_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {app_name} {version}: {e}")
        return {}

def get_app_metadata(app_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Download and parse the CLAMS app directory from GitHub.
    
    Args:
        app_name: Optional name of a specific app to get metadata for
        
    Returns:
        Dictionary mapping app names to their metadata
    """
    # GitHub raw content URL for app-index.json
    url = "https://raw.githubusercontent.com/clamsproject/apps/main/docs/_data/app-index.json"
    
    try:
        logger.info(f"Downloading app directory from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        app_directory = response.json()
        logger.info(f"Successfully loaded {len(app_directory)} apps from GitHub")
        
        # Convert the app directory into the expected format
        formatted_apps = {}
        for app_url, app_info in app_directory.items():
            # Get the latest version
            latest_version = app_info["versions"][0][0] if app_info["versions"] else "unknown"
            
            # Extract app name from URL
            app_name_from_url = app_url.split('/')[-1]
            
            # Fetch detailed metadata for the latest version
            detailed_metadata = fetch_app_metadata(app_name_from_url, latest_version)
            
            formatted_apps[app_name_from_url] = {
                "latest_version": latest_version,
                "metadata": {
                    "description": app_info.get("description", ""),
                    "input": detailed_metadata.get("input", []),
                    "output": detailed_metadata.get("output", []),
                    "parameters": detailed_metadata.get("parameters", [])
                }
            }
            
            logger.info(f"Processed {app_name_from_url} (v{latest_version})")
        
        # Return specific app if requested
        if app_name:
            return formatted_apps.get(app_name, {})
            
        return formatted_apps
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading app directory: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing app directory JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    # When run directly, download the app directory and save it to a file
    output_path = os.path.join(os.path.dirname(__file__), '../data/app_directory.json')
    
    # Add command line argument parsing for forcing fresh download
    import argparse
    parser = argparse.ArgumentParser(description='Download CLAMS app directory')
    parser.add_argument('--force', action='store_true', help='Force fresh download instead of using cache')
    args = parser.parse_args()
    
    app_directory = get_app_metadata()
    print(f"\nRetrieved information for {len(app_directory)} CLAMS apps")
    
    # Print the full formatted app directory
    print("\nFull app directory data:")
    print(json.dumps(app_directory, indent=2))
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(app_directory, f, indent=2)
    print(f"\nSaved app directory to {output_path}")
