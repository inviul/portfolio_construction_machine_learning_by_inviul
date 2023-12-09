from utilities import Scraper as sc
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class Helper():

    def __init__(self) -> None:
         pass

    def createDictDatasetWithFundamentalData(self, tickerArray):
        fTicketDict = dict()
        threads = []
        for ticker in tickerArray:
            scrap = sc.Scraper()
            scrap.setName(name=ticker)
            print(f"Scraping fundamental data for : {scrap.getName()}")
            data = scrap.scrapeFundamentalData(ticker=ticker)
            fTicketDict[ticker]=data
            threads.append(scrap)
            scrap.start()
            scrap.join()
        # for t in threads:
        #     t.join()
        return fTicketDict

    def createDictDatasetWithFundamentalDataUsingPool(self, poolSize, tickerArray):
        print(f"Pool size: {poolSize}")
        scrap = sc.Scraper()
        with ThreadPoolExecutor(max_workers=poolSize) as executor:
            results = executor.map(scrap.scrapeFundamentalData, tickerArray)
        executor.shutdown(wait=True, cancel_futures=False)
        return results
    
    # def createDictDatasetWithFundamentalDataUsingPool(self, poolSize, tickerArray):
    #     fTicketDict = dict()
    #     executor = ThreadPoolExecutor(max_workers=poolSize)
    #     scrap = sc.Scraper()
    #     counter = 1
    #     for ticker in tickerArray:
    #         scrap.setName(name=ticker)
    #         taskName=f"{ticker}_task"
    #         print(f"{counter}. Scraping fundamental data for ---> {scrap.getName()}")
    #         data = scrap.scrapeFundamentalData(ticker=ticker)
    #         taskName = executor.submit(data)
    #         fTicketDict[ticker]=data
    #         counter+=1
    #     return fTicketDict
    
    def StockReturnsComputing(StockPrice, Rows, Columns):    
        StockReturn = np.zeros([Rows-1, Columns])
        for j in range(Columns):        # j: Assets
            for i in range(Rows-1):     # i: Daily Prices
                StockReturn[i,j]=((StockPrice.iloc[i+1, j]-StockPrice.iloc[i,j])/StockPrice.iloc[i,j])* 100

        return StockReturn

