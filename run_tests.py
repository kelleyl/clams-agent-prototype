#!/usr/bin/env python3
"""
Simple test runner for CLAMS Agent conversation quality.
Run this to validate that the agent provides reasonable recommendations.
"""

import requests
import json
import time

# Test cases from test_prompts.md
TEST_CASES = [
    {
        "name": "Basic OCR",
        "prompt": "I want to extract text from video",
        "expected_tools": ["easyocr-wrapper", "tesseractocr-wrapper", "doctr-wrapper", "swt-detection"],
        "should_contain": ["text", "OCR"],
        "should_not_contain": ["transnet-wrapper", "code", "{", "}"]
    },
    {
        "name": "Chyron Text Extraction", 
        "prompt": "extract text from chyrons",
        "expected_tools": ["chyron-detection", "easyocr-wrapper", "swt-detection"],
        "should_contain": ["chyron", "news", "ticker"],
        "should_not_contain": ["transnet-wrapper", "code", "{", "}"]
    },
    {
        "name": "Speech Transcription",
        "prompt": "transcribe speech from video", 
        "expected_tools": ["whisper-wrapper", "distil-whisper-wrapper"],
        "should_contain": ["speech", "transcr"],
        "should_not_contain": ["OCR", "text detection", "code", "{", "}"]
    },
    {
        "name": "Scene Detection",
        "prompt": "detect scene changes in video",
        "expected_tools": ["pyscenedetect-wrapper"],
        "should_contain": ["scene", "detect"],
        "should_not_contain": ["OCR", "speech", "code", "{", "}"]
    },
    {
        "name": "Video Analysis",
        "prompt": "describe what's happening in video frames",
        "expected_tools": ["llava-captioner", "pyscenedetect-wrapper"],
        "should_contain": ["descr", "caption", "image"],
        "should_not_contain": ["OCR", "chyron", "code", "{", "}"]
    }
]

def test_agent_response(prompt, base_url="http://localhost:5000"):
    """Send a test prompt to the agent and return the response."""
    try:
        # Create a session
        session_id = f"test-{int(time.time())}"
        
        # Send the message
        response = requests.post(
            f"{base_url}/api/agui/events",
            json={
                "type": "user_message",
                "data": {"message": prompt},
                "session_id": session_id
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "events" in data and data["events"]:
                for event in data["events"]:
                    if event.get("type") == "text_message_content":
                        content = event.get("data", {})
                        if isinstance(content, dict):
                            response_text = content.get("content", content.get("message", ""))
                            # Skip the initial greeting, look for actual response
                            if "Hello! I'm your CLAMS pipeline assistant" not in response_text:
                                return response_text
                        elif isinstance(content, str):
                            if "Hello! I'm your CLAMS pipeline assistant" not in content:
                                return content
            return "No assistant response found"
        else:
            return f"HTTP Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_response(response, test_case):
    """Evaluate if a response meets the test case criteria."""
    results = {
        "passed": True,
        "issues": [],
        "response": response
    }
    
    # Check for expected tools
    found_tools = []
    for tool in test_case["expected_tools"]:
        if tool in response.lower():
            found_tools.append(tool)
    
    if not found_tools:
        results["passed"] = False
        results["issues"].append(f"No expected tools found. Expected one of: {test_case['expected_tools']}")
    
    # Check should_contain
    for phrase in test_case["should_contain"]:
        if phrase.lower() not in response.lower():
            results["passed"] = False
            results["issues"].append(f"Missing expected phrase: '{phrase}'")
    
    # Check should_not_contain
    for phrase in test_case["should_not_contain"]:
        if phrase.lower() in response.lower():
            results["passed"] = False
            results["issues"].append(f"Contains unwanted phrase: '{phrase}'")
    
    results["found_tools"] = found_tools
    return results

def run_tests():
    """Run all test cases and report results."""
    print("🧪 Running CLAMS Agent Tests...")
    print("=" * 50)
    
    total_tests = len(TEST_CASES)
    passed_tests = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n📝 Test {i}/{total_tests}: {test_case['name']}")
        print(f"Prompt: \"{test_case['prompt']}\"")
        
        # Get response
        response = test_agent_response(test_case["prompt"])
        
        # Evaluate
        result = evaluate_response(response, test_case)
        
        if result["passed"]:
            print("✅ PASSED")
            passed_tests += 1
        else:
            print("❌ FAILED")
            for issue in result["issues"]:
                print(f"   - {issue}")
        
        print(f"Found tools: {result['found_tools']}")
        print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)