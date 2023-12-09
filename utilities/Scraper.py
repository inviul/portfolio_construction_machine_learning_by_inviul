import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import chromedriver_autoinstaller


class Scraper(threading.Thread):
    counter = 1

    def __init__(self) -> None:
         super().__init__()
         chromedriver_autoinstaller.install()

    def testRun(self, tName):
         print(f"Started thread {tName}")
     
    
    def scrapeFundamentalData(self, ticker):
            print(f"{Scraper.counter}. Scraping fundamental data for ---> {ticker}\n")
            opDict = dict()
          #   options = webdriver.ChromeOptions()
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.set_capability('unhandledPromptBehavior', 'accept')
            options.set_capability('pageLoadStrategy', 'eager')
            options.add_argument("--disable-extensions")
            driver = webdriver.Chrome(options=options)
            # driver = webdriver.Chrome()
            driver.get(f'https://finance.yahoo.com/quote/{ticker}/key-statistics?p={ticker}')
            blocks = "(//div[@class='Mstart(a) Mend(a)']/child::div)"
            noOfBlocks = len(blocks)
            lst_n = list()
            for b in range(noOfBlocks-1):
                ele = driver.find_elements(By.XPATH, f"(//div[@class='Mstart(a) Mend(a)']/child::div)[{b}]/child::div/div/div/div/table/tbody/tr/td")
                for i in ele:
                    lst_n.append(i.text)
            lgn = len(lst_n)
            lstn_1 = [lst_n[i] for i in range(0, lgn, 2)]
            lstn_2 = [lst_n[i] for i in range(1, lgn, 2)]
            fundmentatlDataDict = dict(zip(lstn_1, lstn_2))
            opDict[ticker]=fundmentatlDataDict
            driver.close()
            print(f"{Scraper.counter}. Scrapped fundamental data for ---> {ticker}\n")
            Scraper.counter+=1
            return opDict


# if __name__=="__main__":
#     from concurrent.futures import ThreadPoolExecutor
#     tAr = ['ALB', 'APD', 'CE', 'CF', 'CTVA', 'DD', 'DOW', 'ECL', 'EMN', 'FCX', 'FMC', 'IFF', 'LIN', 'LYB', 'MLM', 'MOS', 'NEM', 'NUE', 'PPG', 'SHW','STLD', 'VMC', 'ROP']
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         results = executor.map(Scraper().scrapeFundamentalData, tAr)
        
#     for r in results:
#          print(r)
        

