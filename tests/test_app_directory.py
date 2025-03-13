import unittest
from unittest.mock import patch, MagicMock
import json
from utils.download_app_directory import get_app_metadata, fetch_app_metadata

class TestAppDirectory(unittest.TestCase):
    """Test cases for app directory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample app directory data that mimics the GitHub JSON structure
        self.sample_app_directory = {
            "http://apps.clams.ai/swt-detection": {
                "description": "Detects scenes with text, like slates, chyrons and credits.",
                "latest_update": "2025-02-24T10:41:50+00:00",
                "versions": [
                    ["v7.5", "keighrim"],
                    ["v7.4", "keighrim"]
                ]
            },
            "http://apps.clams.ai/chyron-detection": {
                "description": "This tool detects chyrons, generates time segments.",
                "latest_update": "2023-07-24T21:50:08+00:00",
                "versions": [
                    ["v1.0", "keighrim"]
                ]
            }
        }
        
        # Sample detailed metadata for an app
        self.sample_detailed_metadata = {
            "name": "Scenes-with-text Detection",
            "description": "Detects scenes with text, like slates, chyrons and credits.",
            "app_version": "v7.5",
            "mmif_version": "1.0.5",
            "input": [
                {
                    "@type": "http://mmif.clams.ai/vocabulary/VideoDocument/v1",
                    "required": True
                }
            ],
            "output": [
                {
                    "@type": "http://mmif.clams.ai/vocabulary/TimeFrame/v5",
                    "properties": {
                        "timeUnit": "milliseconds"
                    }
                }
            ],
            "parameters": [
                {
                    "name": "useClassifier",
                    "description": "Use the image classifier model",
                    "type": "boolean",
                    "default": True
                }
            ]
        }
    
    @patch('requests.get')
    def test_fetch_app_metadata(self, mock_get):
        """Test fetching detailed metadata for a specific app version."""
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_detailed_metadata
        mock_get.return_value = mock_response
        
        # Test fetching metadata
        result = fetch_app_metadata('swt-detection', 'v7.5')
        
        # Verify the result
        self.assertEqual(result['name'], "Scenes-with-text Detection")
        self.assertEqual(result['app_version'], "v7.5")
        self.assertEqual(len(result['input']), 1)
        self.assertEqual(len(result['output']), 1)
        self.assertEqual(len(result['parameters']), 1)
        
        # Verify the URL was constructed correctly
        mock_get.assert_called_once_with(
            "https://apps.clams.ai/swt-detection/v7.5/metadata.json"
        )
    
    @patch('requests.get')
    def test_fetch_app_metadata_error(self, mock_get):
        """Test handling of errors when fetching app metadata."""
        # Configure the mock to raise an exception
        mock_get.side_effect = Exception("Network error")
        
        # Test error handling
        result = fetch_app_metadata('swt-detection', 'v7.5')
        self.assertEqual(result, {})
    
    @patch('requests.get')
    def test_get_app_metadata_all_apps(self, mock_get):
        """Test getting metadata for all apps."""
        # Configure the mock responses
        mock_response = MagicMock()
        mock_response.json.side_effect = [
            self.sample_app_directory,  # First call returns app directory
            self.sample_detailed_metadata,  # Second call returns detailed metadata for first app
            self.sample_detailed_metadata  # Third call returns detailed metadata for second app
        ]
        mock_get.return_value = mock_response
        
        # Get all app metadata
        result = get_app_metadata()
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        
        # Check first app
        self.assertIn('swt-detection', result)
        swt_info = result['swt-detection']
        self.assertEqual(swt_info['latest_version'], 'v7.5')
        self.assertEqual(swt_info['metadata']['description'], 
                        "Detects scenes with text, like slates, chyrons and credits.")
        self.assertEqual(len(swt_info['metadata']['input']), 1)
        self.assertEqual(len(swt_info['metadata']['output']), 1)
        self.assertEqual(len(swt_info['metadata']['parameters']), 1)
        
        # Check second app
        self.assertIn('chyron-detection', result)
        chyron_info = result['chyron-detection']
        self.assertEqual(chyron_info['latest_version'], 'v1.0')
        self.assertEqual(chyron_info['metadata']['description'], 
                        "This tool detects chyrons, generates time segments.")
        self.assertEqual(len(chyron_info['metadata']['input']), 1)
        self.assertEqual(len(chyron_info['metadata']['output']), 1)
        self.assertEqual(len(chyron_info['metadata']['parameters']), 1)
    
    @patch('requests.get')
    def test_get_app_metadata_single_app(self, mock_get):
        """Test getting metadata for a specific app."""
        # Configure the mock responses
        mock_response = MagicMock()
        mock_response.json.side_effect = [
            self.sample_app_directory,  # First call returns app directory
            self.sample_detailed_metadata  # Second call returns detailed metadata
        ]
        mock_get.return_value = mock_response
        
        # Get metadata for a specific app
        result = get_app_metadata('swt-detection')
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['latest_version'], 'v7.5')
        self.assertEqual(result['metadata']['description'], 
                        "Detects scenes with text, like slates, chyrons and credits.")
        self.assertEqual(len(result['metadata']['input']), 1)
        self.assertEqual(len(result['metadata']['output']), 1)
        self.assertEqual(len(result['metadata']['parameters']), 1)
    
    @patch('requests.get')
    def test_get_app_metadata_nonexistent_app(self, mock_get):
        """Test getting metadata for a nonexistent app."""
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_app_directory
        mock_get.return_value = mock_response
        
        # Get metadata for a nonexistent app
        result = get_app_metadata('nonexistent-app')
        
        # Verify the result is an empty dict
        self.assertEqual(result, {})
    
    @patch('requests.get')
    def test_get_app_metadata_request_error(self, mock_get):
        """Test handling of request errors."""
        # Configure the mock to raise an exception
        mock_get.side_effect = Exception("Network error")
        
        # Verify that the exception is raised
        with self.assertRaises(Exception):
            get_app_metadata()
    
    @patch('requests.get')
    def test_get_app_metadata_invalid_json(self, mock_get):
        """Test handling of invalid JSON response."""
        # Configure the mock to return invalid JSON
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response
        
        # Verify that the JSONDecodeError is raised
        with self.assertRaises(json.JSONDecodeError):
            get_app_metadata()

if __name__ == '__main__':
    unittest.main() 