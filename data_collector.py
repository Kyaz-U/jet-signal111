import random
import time

def get_latest_coefficients():
    """
    Simulates scraping live Aviator coefficients from 1win/Mostbet.
    In production, replace this with requests + BeautifulSoup or Selenium logic.
    """
    # Simulate a 1-second delay as if requesting real-time data
    time.sleep(1)

    # Simulate a dynamic list of the latest 5 coefficients (mocked)
    latest_data = [round(random.uniform(1.0, 10.0), 2) for _ in range(5)]
    return latest_data