"""Standalone test for robots.txt parsing logic"""

def check_robots_txt_logic(robots_text):
    """Test the robots.txt parsing logic without dependencies"""
    robots_content = robots_text.lower()
    
    # Check for exact "Disallow: /" pattern (not "Disallow: /admin" etc.)
    # Look for "disallow: /" followed by whitespace or newline
    if "user-agent: *" in robots_content:
        lines = robots_content.split('\n')
        for line in lines:
            line = line.strip()
            # Check if line is exactly "disallow: /" or "disallow: /" followed by whitespace
            if line.startswith("disallow:"):
                path = line.replace("disallow:", "").strip()
                if path == "/":
                    return {
                        "allowed": False,
                        "warning": "This website's robots.txt discourages scraping. Please respect their wishes.",
                    }
    return {"allowed": True}


# Test case 1: robots.txt that allows scraping (only disallows /admin)
print("Test 1 - Disallow /admin only:")
robots_text_1 = "User-agent: *\nDisallow: /admin"
result_1 = check_robots_txt_logic(robots_text_1)
print(f"  Input: {repr(robots_text_1)}")
print(f"  Result: {result_1}")
print(f"  Expected: allowed=True")
test1_pass = result_1['allowed'] == True
print(f"  PASS: {test1_pass}\n")

# Test case 2: robots.txt that disallows scraping (disallows /)
print("Test 2 - Disallow / (everything):")
robots_text_2 = "User-agent: *\nDisallow: /"
result_2 = check_robots_txt_logic(robots_text_2)
print(f"  Input: {repr(robots_text_2)}")
print(f"  Result: {result_2}")
print(f"  Expected: allowed=False")
test2_pass = result_2['allowed'] == False
print(f"  PASS: {test2_pass}\n")

# Test case 3: robots.txt not found
print("Test 3 - No user-agent match:")
robots_text_3 = "User-agent: Googlebot\nDisallow: /"
result_3 = check_robots_txt_logic(robots_text_3)
print(f"  Input: {repr(robots_text_3)}")
print(f"  Result: {result_3}")
print(f"  Expected: allowed=True (no user-agent: * match)")
test3_pass = result_3['allowed'] == True
print(f"  PASS: {test3_pass}\n")

if test1_pass and test2_pass and test3_pass:
    print("✓ All tests passed!")
    exit(0)
else:
    print("✗ Some tests failed!")
    exit(1)
