import requests
from bs4 import BeautifulSoup
import json
import os
from typing import Dict, List, Any, Optional
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
        response = requests.get(metadata_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {app_name} {version}: {e}")
        return {}

def download_app_directory(output_file: Optional[str] = None, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Downloads and parses the CLAMS app directory from https://apps.clams.ai/
    
    Args:
        output_file: Optional path to save the JSON output
        use_cache: If True and output_file exists, load from cache instead of scraping
        
    Returns:
        Dictionary mapping app names to their metadata (description, latest version)
    """
    # Check for cache file if use_cache is True
    if use_cache and output_file and os.path.exists(output_file):
        try:
            logger.info(f"Loading app directory from cache file: {output_file}")
            with open(output_file, 'r') as f:
                app_directory = json.load(f)
            logger.info(f"Successfully loaded {len(app_directory)} apps from cache")
            return app_directory
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}, will scrape instead")
    
    url = "https://apps.clams.ai/"
    logger.info(f"Downloading CLAMS app directory from {url}")
    
    try:
        # Download the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Dictionary to store app metadata
        app_directory: Dict[str, Dict[str, Any]] = {}
        
        # Find all app sections (each app is under an h3 tag)
        app_headers = soup.find_all('h3')
        
        for header in app_headers:
            # The app name is the text content of the h3 tag
            app_name = header.get_text().strip()
            
            # Skip if not an app header
            if not app_name or app_name == "CLAMS Team" or "On this page" in app_name:
                continue
                
            # The description is in the paragraph following the h3 tag
            description_elem = header.find_next('p')
            description = description_elem.get_text().strip() if description_elem else "No description available"
            
            # Version info is in a list following the description
            versions = []
            version_list = header.find_next('ul')
            if version_list:
                for item in version_list.find_all('li'):
                    version_text = item.get_text().strip()
                    # Split by space and extract version number
                    if version_text.startswith('v'):
                        version_parts = version_text.split(' ')
                        if len(version_parts) > 0:
                            version = version_parts[0].strip()
                            versions.append(version)
            
            # Get the latest version (first in the list)
            latest_version = versions[0] if versions else "unknown"
            
            # Create base app metadata
            app_metadata = {
                "name": app_name,
                "description": description,
                "latest_version": latest_version,
                "all_versions": versions,
                "metadata": {}
            }
            
            # Fetch detailed metadata for latest version if available
            if latest_version != "unknown":
                logger.info(f"Fetching metadata for {app_name} {latest_version}")
                metadata = fetch_app_metadata(app_name.lower(), latest_version)
                if metadata:
                    # Extract relevant metadata fields
                    app_metadata["metadata"] = {
                        "description": metadata.get("description", ""),
                        "input": metadata.get("input", []),
                        "output": metadata.get("output", []),
                        "parameters": metadata.get("parameters", []),
                        "app_version": metadata.get("app_version", ""),
                        "mmif_version": metadata.get("mmif_version", ""),
                        "identifier": metadata.get("identifier", ""),
                        "url": metadata.get("url", "")
                    }
            
            # Store the metadata
            app_directory[app_name] = app_metadata
        
        logger.info(f"Successfully parsed {len(app_directory)} CLAMS apps")
        
        # Save to file if output_file is specified
        if output_file:
            directory = os.path.dirname(output_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(output_file, 'w') as f:
                json.dump(app_directory, f, indent=2)
            logger.info(f"Saved app directory to {output_file}")
        
        return app_directory
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading app directory: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing app directory: {e}")
        return {}

def get_app_metadata(app_name: str = None) -> Dict[str, Any]:
    """
    Gets metadata for a specific app or all apps
    
    Args:
        app_name: Name of the app to get metadata for. If None, returns all apps.
        
    Returns:
        Dictionary containing the app metadata or all app metadata
    """
    cache_file = os.path.join(os.path.dirname(__file__), '../data/app_directory.json')
    logger.debug(f"Loading app metadata from cache file: {cache_file}")
    
    # Use download_app_directory with cache option
    app_directory = download_app_directory(output_file=cache_file, use_cache=True)
    logger.debug(f"App directory type: {type(app_directory)}")
    logger.debug(f"App directory keys: {list(app_directory.keys()) if isinstance(app_directory, dict) else 'Not a dictionary'}")
    
    if app_name:
        return app_directory.get(app_name, {})
    return app_directory

def get_app_metadata() -> Dict[str, Any]:
    """
    Download and parse the CLAMS app directory from GitHub.
    
    Returns:
        Dictionary mapping app URLs to their metadata
    """
    # GitHub raw content URL for app-index.json
    # Using raw.githubusercontent.com instead of github.com/blob/ to get the raw file
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
            
            formatted_apps[app_url] = {
                "latest_version": latest_version,
                "metadata": {
                    "description": app_info.get("description", ""),
                    "input": [],  # These will be populated by the pipeline generator as needed
                    "output": []
                }
            }
        
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
    
    app_directory = download_app_directory(output_path, use_cache=not args.force)
    print(f"Retrieved information for {len(app_directory)} CLAMS apps")
    
    # Display some sample data
    if app_directory:
        sample_app = next(iter(app_directory.values()))
        print(f"\nSample app data:")
        print(json.dumps(sample_app, indent=2))
