import requests

# IMPORTANT: put your real name and email. The SEC blocks requests without one.
HEADERS = {"User-Agent": "Your Name your.email@gmail.com"}

url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000829224.json"
r = requests.get(url, headers=HEADERS, timeout=30)
print("Status code:", r.status_code)   # 200 means success

data = r.json()
print("Company:", data["entityName"])
print("Revenue tags available:",
      [t for t in data["facts"]["us-gaap"] if "Revenue" in t][:10])