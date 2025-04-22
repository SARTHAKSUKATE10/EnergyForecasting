from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import openpyxl

# Set up ChromeDriver with webdriver-manager (automatically downloads the correct driver)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open your local or hosted app
driver.get("http://127.0.0.1:5000")  # Change this URL if needed

# Wait for the page to load completely
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "date"))
)

# Test cases data (add more as needed)
test_cases = [
    {
        "date": "2022-06-01", "season": "Summer", "period": "Post-lockdown",
        "temp": "32", "rainfall": "0", "humidity": "43", "population": "4500000", "festival": "0"
    },
    {
        "date": "2022-06-02", "season": "Winter", "period": "Pre-lockdown",
        "temp": "10", "rainfall": "5", "humidity": "80", "population": "4000000", "festival": "1"
    },
    # Add more test cases as needed
]

# Set up Excel file
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["Date", "Season", "Period", "Temp", "Rainfall", "Humidity", "Population", "Festival", "Prediction"])

# Loop through test cases and run each one
for case in test_cases:
    # Fill out the form
    driver.find_element(By.ID, "date").send_keys(case["date"])
    driver.find_element(By.ID, "season").send_keys(case["season"])
    driver.find_element(By.ID, "period").send_keys(case["period"])
    driver.find_element(By.ID, "temp").send_keys(case["temp"])
    driver.find_element(By.ID, "rainfall").send_keys(case["rainfall"])
    driver.find_element(By.ID, "humidity").send_keys(case["humidity"])
    driver.find_element(By.ID, "population").send_keys(case["population"])
    driver.find_element(By.ID, "Festival").send_keys(case["festival"])

    # Submit the form
    submit_button = driver.find_element(By.CSS_SELECTOR, "form#predictionForm button[type='submit']")
    submit_button.click()

    # Wait for the prediction result to appear
    output = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "predictionOutput"))
    )

    # Get the result text
    prediction = output.text

    # Write the results to Excel
    ws.append([case["date"], case["season"], case["period"], case["temp"], case["rainfall"],
               case["humidity"], case["population"], case["festival"], prediction])

    # Clear form for next test case
    driver.find_element(By.ID, "date").clear()
    driver.find_element(By.ID, "season").clear()
    driver.find_element(By.ID, "period").clear()
    driver.find_element(By.ID, "temp").clear()
    driver.find_element(By.ID, "rainfall").clear()
    driver.find_element(By.ID, "humidity").clear()
    driver.find_element(By.ID, "population").clear()
    driver.find_element(By.ID, "Festival").clear()

    # Adding a delay so the browser can finish processing
    time.sleep(2)

# Save the Excel file
wb.save("test_results.xlsx")

# Close the browser
driver.quit()

print("Test cases executed and results saved to 'test_results.xlsx'")
