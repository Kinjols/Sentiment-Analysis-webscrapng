import sys
import csv
from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# accepting cookies
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC



# default path to file to store data
path_to_file = "reviews1.csv"

# default number of scraped pages
num_page = 20

url=  "https://www.tripadvisor.com/Hotel_Review-g188590-d190614-Reviews-InterContinental_Amstel_Amsterdam-Amsterdam_North_Holland_Province.html"

driver = webdriver.Firefox(executable_path="geckodriver.exe")

driver.get(url)
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='_evidon-accept-button']"))).click()

scrapedReviews=[]

# change the value inside the range to save more or less reviews
for i in range(0, num_page):

    # expand the review 
    time.sleep(2)
    #driver.find_element_by_xpath(".//div[contains(@data-test-target, 'expand-review')]").click()

    container = driver.find_elements_by_xpath("//div[@data-reviewid]")


    for j in range(len(container)):

        rating = container[j].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        title = container[j].find_element_by_xpath(".//div[contains(@data-test-target, 'review-title')]").text
        review = container[j].find_element_by_xpath(".//div[contains(@class,'pIRBV')]").text.replace("\n", "  ")
            
        scrapedReviews.append([title,review,rating]) 
        
    # change the page            
    driver.find_element_by_xpath('.//a[@class="ui_button nav next primary "]').click()


scrapedReviewsDF = pd.DataFrame(scrapedReviews, columns=['title', 'review', 'rating'])
scrapedReviewsDF.head(10)
# driver.quit()
print( 'Ready scraping ....')
scrapedReviewsDF.to_csv("trip_advisor_review.csv", sep=',',index= False)
