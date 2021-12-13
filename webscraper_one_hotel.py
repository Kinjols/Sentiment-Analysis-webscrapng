import pandas as pd
import time
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common import by
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# going to the website with the webdriver
driver = webdriver.Firefox(executable_path="geckodriver.exe")
driver.get('https://www.booking.com/hotel/nl/park-inn-by-radisson-amsterdam-city-west.en-gb.html?aid=304142;label=gen173nr-1FCBcoggI46AdIM1gEaKkBiAEBmAEJuAEXyAEP2AEB6AEB-AECiAIBqAIDuAKzldqLBsACAdICJDNhY2UwMjFkLWQ4MTYtNGYyMS1iMjk2LTIxODIwNDc1ZTY4OdgCBeACAQ;sid=7ba796949fabe9953261480dff3918ad;dest_id=-2140479;dest_type=city;dist=0;group_adults=2;group_children=0;hapos=0;hpos=0;no_rooms=1;req_adults=2;req_children=0;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1635179469;srpvid=17c67425457b00b0;type=total;ucfs=1;sig=v1SlkxZCZN&#tab-reviews')

# array in which we will store the scraped reviews
scrapedReviews=[]

# locate cookkie handler and click it away
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='onetrust-accept-btn-handler']"))).click()

# amount of pages we will collect reviews from
num_page = 10

for i in range(0, num_page):

    # find the reviews by html/xpath ID
    time.sleep(2)
    reviews = driver.find_elements_by_xpath("//*[@class='bui-grid__column-9 c-review-block__right']")

    # collect each individual review
    for i in range(len(reviews)):
        title = reviews[i].find_element_by_xpath(".//div[@class='bui-grid__column-10']").text
        scraped_dubbel_review = reviews[i].find_element_by_xpath(".//div[@class='c-review']").text.replace("\n", " ").replace("Liked", " ")
        score = reviews[i].find_element_by_xpath(".//div[@class='bui-review-score__badge']").text
        
        # split review into negative ans positive part
        scraped_dubbel_review = scraped_dubbel_review.split("Disliked")

        # determine the sentiment of the review parts 
        for i in range(len(scraped_dubbel_review)):
            review = scraped_dubbel_review[i]
            if((scraped_dubbel_review[i]!="") and (i==0)):
                scrapedReviews.append([title,review,score, 1,"booking"])
            elif((scraped_dubbel_review[i]!="") and (i==1)):
                scrapedReviews.append([title,review,score, 0,"booking"])
                
    # go to the next page of reviews till specified amount of pages is reached 
    nextPage = driver.find_element_by_class_name('bui-pagination__next-arrow')     
    nextPage.click()            
    

# put scraped review into a dataframe
scrapedReviewsDF = pd.DataFrame(scrapedReviews, columns=['title', 'review', 'rating','sentiment','source'])
driver.quit()
scrapedReviewsDF.head(60)
print( 'Ready scraping ....')
scrapedReviewsDF.to_csv("booking_reviews.csv", sep=',',index= False)

