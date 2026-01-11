"""Quick test to verify robots.txt fix"""
import sys
from unittest.mock import Mock, patch

# Mock streamlit before importing backend
sys.modules["streamlit"] = Mock()

from backend import check_robots_txt

# Test case 1: robots.txt that allows scraping (only disallows /admin)
@patch("backend.requests.get")
def test_allows_scraping(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "User-agent: *\nDisallow: /admin"
    mock_get.return_value = mock_response
    
    result = check_robots_txt("https://example.com")
    print(f"Test 1 - Disallow /admin only:")
    print(f"  Result: {result}")
    print(f"  Expected: allowed=True")
    print(f"  PASS: {result['allowed'] == True}\n")
    return result["allowed"] == True

# Test case 2: robots.txt that disallows scraping (disallows /)
@patch("backend.requests.get")
def test_disallows_scraping(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "User-agent: *\nDisallow: /"
    mock_get.return_value = mock_response
    
    result = check_robots_txt("https://example.com")
    print(f"Test 2 - Disallow / (everything):")
    print(f"  Result: {result}")
    print(f"  Expected: allowed=False")
    print(f"  PASS: {result['allowed'] == False}\n")
    return result["allowed"] == False

if __name__ == "__main__":
    test1_pass = test_allows_scraping()
    test2_pass = test_disallows_scraping()
    
    if test1_pass and test2_pass:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
