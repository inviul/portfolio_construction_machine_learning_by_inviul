from django.shortcuts import render, HttpResponse
from portfolioconsapp.utilities import *
import pandas as pd
import mimetypes
import numpy as np
from sklearn.cluster import KMeans


# Create your views here.

budget = 1000
priceData = pd.DataFrame()
all500Tickers = dict()
fDframe = pd.DataFrame()
portfolios = dict()
portfolio1 = pd.DataFrame()
portfolio2 = pd.DataFrame()
X_train1, X_test1, y_train1, y_test1 = 0,0,0,0
X_train2, X_test2, y_train2, y_test2 = 0,0,0,0
columns=['Model Name', 'MeanAbsError', 'MeanSqrError', 'RootMeanSqrError', 'R2SqrError', 'CrossValScore']
modelEvalPortfolio1 = pd.DataFrame(columns=columns)
modelEvalPortfolio2 = pd.DataFrame(columns=columns)


def index(request):
    return render(request, 'index.html')

def downloadfile(request, filename=''):
    if filename != '':
        # Define Django project base directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Define the full file path
        filepath = BASE_DIR + '/downloads/' + filename
        # Open the file for reading content
        path = open(filepath, 'rb')
        # Set the mime type
        mime_type, _ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        response = HttpResponse(path, content_type=mime_type)
        # Set the HTTP header for sending to browser
        response['Content-Disposition'] = "attachment; filename=%s" % filename
        # Return the response value
        return response


def buildDatasets(request):
    userInput = getUserInputs(request)
    global budget
    budget = userInput[1]['budget']

    if not False in userInput[0].values():
        global priceData, all500Tickers, fDframe, portfolios, portfolio1, portfolio2
        priceData = fetchPricingData()
        all500Tickers = fetchFundamentalData()
        fDframe = createDataframeWithFundamentalDict(all500Tickers)
        portfolios = dict()
        portfolio1 = pd.DataFrame()
        portfolio2 = pd.DataFrame()
        emptyColDropList = checkEmptyDataTicker(priceData)
        if len(emptyColDropList) > 0:
            priceData = priceData.drop(emptyColDropList, axis=1)
            fDframe = fDframe.drop(emptyColDropList)
        priceData = priceData.ffill()

        #Optimization Theory
        dailyReturn = np.array(StockReturnsComputing(StockPrice=priceData, Rows=priceData.shape[0], Columns=priceData.shape[1]))
        # compute mean returns and variance covariance matrix of returns
        meanReturns = np.mean(dailyReturn, axis=0)
        np.savetxt("./downloads/meanreturn.csv", meanReturns, delimiter=",")
        covReturns = np.cov(dailyReturn, rowvar=False)
        # set precision for printing results
        np.set_printoptions(precision=5, suppress=True)
        np.savetxt("./downloads/varcovarmatrix.csv", covReturns, delimiter=",")
        meanReturns = meanReturns.reshape(len(meanReturns), 1)
        assetParameters = np.concatenate([meanReturns, covReturns], axis=1)
        np.savetxt("./downloads/assetparameters.csv", assetParameters, delimiter=",")

        #Working on fundamental dataset
        fDframe = fDframe[
            ['Price/Book (mrq)', 'Operating Cash Flow (ttm)', 'Operating Margin (ttm)', 'Revenue Per Share (ttm)',
             'Enterprise Value/Revenue', 'Gross Profit (ttm)', 'PEG Ratio (5 yr expected)',
             'Current Ratio (mrq)', 'Enterprise Value', '50-Day Moving Average 3', 'Diluted EPS (ttm)', 'Revenue (ttm)',
             'Enterprise Value/EBITDA', 'Total Cash Per Share (mrq)', 'EBITDA', 'Beta (5Y Monthly)',
             'Market Cap (intraday)',
             'Forward Annual Dividend Rate 4', 'Book Value Per Share (mrq)', 'Price/Sales (ttm)',
             'Return on Assets (ttm)', 'Profit Margin',
             'Quarterly Revenue Growth (yoy)', 'Forward P/E', 'Float 8', '200-Day Moving Average 3', ]]

        fDframe.columns = fDframe.columns.str.replace(" ", "_")
        colList = list(fDframe.columns)
        fDframe = fDframe[colList].apply(lambda x: x.str.strip()).replace('N/A', np.nan)

        # Identify currency fetures into a list
        currencyFeatures = ['Operating_Cash_Flow_(ttm)', 'Gross_Profit_(ttm)', 'Enterprise_Value', 'Revenue_(ttm)',
                            'EBITDA',
                            'Market_Cap_(intraday)', 'Float_8']
        fDframeCurrency = pd.DataFrame(index=fDframe.index, columns=currencyFeatures)
        for col in currencyFeatures:
            for record in range(fDframe.shape[0]):
                fDframeCurrency.loc[fDframe.index[record]][col] = convertMillionToBillion(
                    str(fDframe.iloc[record][col]))
        # Identify percent fetures into a list
        percentFeatures = ['Operating_Margin_(ttm)', 'Return_on_Assets_(ttm)', 'Profit_Margin',
                           'Quarterly_Revenue_Growth_(yoy)']

        fDframePercent = pd.DataFrame(index=fDframe.index, columns=percentFeatures)

        for col in percentFeatures:
            for record in range(fDframe.shape[0]):
                val = fDframe.iloc[record][col]
                if pd.isnull(val) == False:
                    fDframePercent.loc[fDframe.index[record]][col] = convertPercentToRatio(
                        str(fDframe.iloc[record][col]))
                else:
                    fDframePercent.loc[fDframe.index[record]][col] = np.nan

        featuresToDrop = currencyFeatures + percentFeatures
        fDframeResidual = fDframe.drop(featuresToDrop, axis=1)
        mFundamentalData = pd.merge(fDframeCurrency, fDframePercent, left_index=True, right_index=True)
        fDframeResidualCols = fDframeResidual.columns
        fDframeResd = pd.DataFrame(index=fDframeResidual.index, columns=fDframeResidualCols)

        for col in fDframeResidualCols:
            for record in range(fDframeResidual.shape[0]):
                val = fDframeResidual.iloc[record][col]
                if pd.isnull(val) == False:
                    fDframeResd.loc[fDframeResidual.index[record]][col] = removeRandomChar(
                        str(fDframeResidual.iloc[record][col]))
                else:
                    fDframeResd.loc[fDframeResidual.index[record]][col] = np.nan

        fundatmentalData = pd.merge(mFundamentalData, fDframeResd, left_index=True, right_index=True)
        fundatmentalData.to_csv("./downloads/fundamental.csv")

        #Clustering for Portfolio construction
        clusters = 2
        assetsCluster = KMeans(algorithm='auto', max_iter=600, n_clusters=clusters)
        assetsCluster.fit(assetParameters)
        centroids = assetsCluster.cluster_centers_
        labels = assetsCluster.labels_
        assetLabels = [ticker for ticker in priceData]
        assets = np.array(assetLabels)
        for i in range(clusters):
            clt = np.where(labels == i)
            assetsCluster = assets[clt]
            portfolios[f"portfolio_{i + 1}"] = assetsCluster
        with open('./downloads/portfoliocluster.txt', 'w') as convert_file:
            convert_file.write(str(portfolios))

        # fixing asset labels to cluster points
        print('Stocks in each of the portfolios:\n', )
        assets = np.array(assetLabels)
        portfolios = dict()
        for i in range(clusters):
            clt = np.where(labels == i)
            assetsCluster = assets[clt]
            portfolios[f"portfolio_{i + 1}"] = assetsCluster

        pFolio = dict()
        for k, v in portfolios.items():
            dfName = f"df_{k}"
            dfName = pd.DataFrame()
            for asset in v:
                dfName[asset] = priceData.loc[:, asset]
            pFolio[k] = dfName

        pFolioLog = dict()
        pFolioLog['portfolio_1'] = np.log(pFolio['portfolio_1'])
        pFolioLog['portfolio_2'] = np.log(pFolio['portfolio_2'])

        pFolioLogReturn = dict()
        dframe1 = pFolioLog['portfolio_1']
        getLogReturnOnAsset(dframe1, pFolioLogReturn, "portfolio_1")
        dframe2 = pFolioLog['portfolio_2']
        getLogReturnOnAsset(dframe2, pFolioLogReturn, "portfolio_2")
        pFolioLog['portfolio_1'].to_csv("./downloads/portfolio_1.csv")
        pFolioLog['portfolio_2'].to_csv("./downloads/portfolio_2.csv")
        pFolioLogReturn['portfolio_1'].to_csv("./downloads/portfolio_1_logreturn.csv")
        pFolioLogReturn['portfolio_2'].to_csv("./downloads/portfolio_2_logreturn.csv")
        firstCombinedTren(pFolioLog)

        #Final Dataset
        portfolio1 = pd.merge(fundatmentalData, pFolioLogReturn['portfolio_1'], left_index=True, right_index=True)
        portfolio1 = portfolio1.apply(pd.to_numeric)
        portfolio1 = fillMissingValue(portfolio1)
        portfolio2 = pd.merge(fundatmentalData, pFolioLogReturn['portfolio_2'], left_index=True, right_index=True)
        portfolio2 = portfolio2.apply(pd.to_numeric)
        portfolio2 = fillMissingValue(portfolio2)

        #Feature engineering
        portfolio1 = featureEngineering(fDframe, meanReturns, portfolio1)
        portfolio1.to_csv("./downloads/portfolio1_final.csv")
        portfolio2 = featureEngineering(fDframe, meanReturns, portfolio2)
        portfolio2.to_csv("./downloads/portfolio2_final.csv")

    else:
        print("False hai")

    return render(request, 'result.html', {'result': 'a'})


def generateVisualizationForP1(request):
    downoadSNSPlot(portfolio1, "portfolio1_corr")
    trainTestSplitAndFeatureRanking(portfolio1, "1")
    global X_train1, X_test1, y_train1, y_test1
    X_train1, X_test1, y_train1, y_test1 = trainTestSplitAndFeatureRanking(portfolio1, "1")
    return render(request, 'visual_p1.html')

def generateVisualizationForP2(request):
    downoadSNSPlot(portfolio2, "portfolio2_corr")
    global X_train2, X_test2, y_train2, y_test2
    X_train2, X_test2, y_train2, y_test2 = trainTestSplitAndFeatureRanking(portfolio2, "2")
    return render(request, 'visual_p2.html')

def modelBuilding(portfolio):
    global modelEvalPortfolio1, modelEvalPortfolio2, columns
    columns = ['Model Name', 'MeanAbsError', 'MeanSqrError', 'RootMeanSqrError', 'R2SqrError', 'CrossValScore']
    modelEvalPortfolio1 = pd.DataFrame(columns=columns)
    modelEvalPortfolio2 = pd.DataFrame(columns=columns)
    if portfolio=='portfolio1':
        l1 = linearRegression(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(l1, ignore_index=True)
        d1 = decisionTree(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(d1, ignore_index=True)
        sv1 = supportVectorMachine(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(sv1, ignore_index=True)
        rf1 = randomForestRegressor(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(rf1, ignore_index=True)
        lr1 = lassoRegression(modelEvalPortfolio=modelEvalPortfolio1, alpha=0.0009001, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(lr1, ignore_index=True)
        en1 = elasticNet(modelEvalPortfolio=modelEvalPortfolio1, alpha=0.00000001, l1_ratio=0.9, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(en1, ignore_index=True)
        rr1 = ridgeRegression(modelEvalPortfolio=modelEvalPortfolio1, tol=0.0000001, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(rr1, ignore_index=True)
        sg1 = sgdModel(modelEvalPortfolio=modelEvalPortfolio1, n_iter_no_change=2500, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
        modelEvalPortfolio1 = modelEvalPortfolio1._append(sg1, ignore_index=True)
        an1 = ann1(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1)
        modelEvalPortfolio1 = modelEvalPortfolio1._append(an1, ignore_index=True)
        print(modelEvalPortfolio1)
        dataframToImage(modelEvalPortfolio1,"./static/images/df_1.jpg")
    elif portfolio=='portfolio2':
        l2 = linearRegression(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(l2, ignore_index=True)
        d2 = decisionTree(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(d2, ignore_index=True)
        sv2 = supportVectorMachine(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(sv2, ignore_index=True)
        rf2 = randomForestRegressor(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(rf2, ignore_index=True)
        lr2 = lassoRegression(modelEvalPortfolio=modelEvalPortfolio2, alpha=0.000001, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(lr2, ignore_index=True)
        en2 = elasticNet(modelEvalPortfolio=modelEvalPortfolio2, alpha=0.001, l1_ratio=0.2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(en2, ignore_index=True)
        rr2  = ridgeRegression(modelEvalPortfolio=modelEvalPortfolio2, tol=0.1, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(rr2, ignore_index=True)
        sg2 = sgdModel(modelEvalPortfolio=modelEvalPortfolio2, n_iter_no_change=25000, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2, y1_Te=y_test2, pNo='2')
        modelEvalPortfolio2 = modelEvalPortfolio2._append(sg2, ignore_index=True)
        an2 = ann2(modelEvalPortfolio=modelEvalPortfolio2, x2_Tr=X_train2, y2_Tr=y_train2, x2_Te=X_test2, y2_Te=y_test2)
        modelEvalPortfolio2 = modelEvalPortfolio2._append(an2, ignore_index=True)
        print(modelEvalPortfolio2)
        dataframToImage(modelEvalPortfolio2, "./static/images/df_2.jpg")
    port = portfolio.title()

    return port

def model1(request):
    # port = modelBuilding('portfolio1')
    global modelEvalPortfolio1,  columns
    columns = ['Model Name', 'MeanAbsError', 'MeanSqrError', 'RootMeanSqrError', 'R2SqrError', 'CrossValScore']
    modelEvalPortfolio1 = pd.DataFrame(columns=columns)
    l1 = linearRegression(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1,
                          y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(l1, ignore_index=True)
    d1 = decisionTree(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1,
                      y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(d1, ignore_index=True)
    sv1 = supportVectorMachine(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1,
                               y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(sv1, ignore_index=True)
    rf1 = randomForestRegressor(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1,
                                y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(rf1, ignore_index=True)
    lr1 = lassoRegression(modelEvalPortfolio=modelEvalPortfolio1, alpha=0.0009001, x1_Tr=X_train1, y1_Tr=y_train1,
                          x1_Te=X_test1, y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(lr1, ignore_index=True)
    en1 = elasticNet(modelEvalPortfolio=modelEvalPortfolio1, alpha=0.00000001, l1_ratio=0.9, x1_Tr=X_train1,
                     y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(en1, ignore_index=True)
    rr1 = ridgeRegression(modelEvalPortfolio=modelEvalPortfolio1, tol=0.0000001, x1_Tr=X_train1, y1_Tr=y_train1,
                          x1_Te=X_test1, y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(rr1, ignore_index=True)
    sg1 = sgdModel(modelEvalPortfolio=modelEvalPortfolio1, n_iter_no_change=2500, x1_Tr=X_train1, y1_Tr=y_train1,
                   x1_Te=X_test1, y1_Te=y_test1, pNo='1')
    modelEvalPortfolio1 = modelEvalPortfolio1._append(sg1, ignore_index=True)
    an1 = ann1(modelEvalPortfolio=modelEvalPortfolio1, x1_Tr=X_train1, y1_Tr=y_train1, x1_Te=X_test1, y1_Te=y_test1)
    modelEvalPortfolio1 = modelEvalPortfolio1._append(an1, ignore_index=True)
    print(modelEvalPortfolio1)
    dataframToImage(modelEvalPortfolio1, "./static/images/df_1.jpg")
    return render(request, 'model.html', {'port': 'Portfolio1'})

def model2(request):
    # port = modelBuilding('portfolio2')
    global modelEvalPortfolio2, columns
    columns = ['Model Name', 'MeanAbsError', 'MeanSqrError', 'RootMeanSqrError', 'R2SqrError', 'CrossValScore']
    modelEvalPortfolio2 = pd.DataFrame(columns=columns)
    l2 = linearRegression(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2,
                          y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(l2, ignore_index=True)
    d2 = decisionTree(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2,
                      y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(d2, ignore_index=True)
    sv2 = supportVectorMachine(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2,
                               y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(sv2, ignore_index=True)
    rf2 = randomForestRegressor(modelEvalPortfolio=modelEvalPortfolio2, x1_Tr=X_train2, y1_Tr=y_train2, x1_Te=X_test2,
                                y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(rf2, ignore_index=True)
    lr2 = lassoRegression(modelEvalPortfolio=modelEvalPortfolio2, alpha=0.000001, x1_Tr=X_train2, y1_Tr=y_train2,
                          x1_Te=X_test2, y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(lr2, ignore_index=True)
    en2 = elasticNet(modelEvalPortfolio=modelEvalPortfolio2, alpha=0.001, l1_ratio=0.2, x1_Tr=X_train2, y1_Tr=y_train2,
                     x1_Te=X_test2, y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(en2, ignore_index=True)
    rr2 = ridgeRegression(modelEvalPortfolio=modelEvalPortfolio2, tol=0.1, x1_Tr=X_train2, y1_Tr=y_train2,
                          x1_Te=X_test2, y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(rr2, ignore_index=True)
    sg2 = sgdModel(modelEvalPortfolio=modelEvalPortfolio2, n_iter_no_change=25000, x1_Tr=X_train2, y1_Tr=y_train2,
                   x1_Te=X_test2, y1_Te=y_test2, pNo='2')
    modelEvalPortfolio2 = modelEvalPortfolio2._append(sg2, ignore_index=True)
    an2 = ann2(modelEvalPortfolio=modelEvalPortfolio2, x2_Tr=X_train2, y2_Tr=y_train2, x2_Te=X_test2, y2_Te=y_test2)
    modelEvalPortfolio2 = modelEvalPortfolio2._append(an2, ignore_index=True)
    print(modelEvalPortfolio2)
    dataframToImage(modelEvalPortfolio2, "./static/images/df_2.jpg")
    return render(request, 'model2.html', {'port': 'Portfolio2'})

def recommendation(request):
    msg = ""
    print('List1- ', modelEvalPortfolio1['R2SqrError'])
    print('List2- ', modelEvalPortfolio2['R2SqrError'])
    r2Score1 = modelEvalPortfolio1['R2SqrError'].sort_values()
    print('l- ',r2Score1)
    r2Score1Sorted = list(r2Score1)
    print('sorted- ', r2Score1Sorted)
    r2Score2 = modelEvalPortfolio2['R2SqrError'].sort_values()
    print('l- ', r2Score2)
    r2Score2Sorted = list(r2Score2)
    print('sorted- ', r2Score2Sorted)
    r2Score1Len = len(r2Score1)
    r2Score2Len = len(r2Score2)
    print(r2Score1Len)
    print(r2Score2Len)
    highIndex1 = r2Score1Sorted[r2Score1Len-1]
    highIndex2 = r2Score2Sorted[r2Score2Len - 1]

    print(highIndex1)
    print(highIndex2)
    if highIndex1>highIndex2:
        msg = "Portfolio 1 has better model score!"
        maxExpPortfolioReturn2, minRiskExpPortfolioReturn2, allocation, fundLeft  = returnCalculator(portfolio1, priceData, budget)
    else:
        msg = "Portfolio 2 has better model score!"
        maxExpPortfolioReturn2, minRiskExpPortfolioReturn2, allocation, fundLeft  = returnCalculator(portfolio2, priceData, budget)
    print(r2Score2Len, r2Score1Len, highIndex1, highIndex2, maxExpPortfolioReturn2, minRiskExpPortfolioReturn2, msg)
    return render(request, 'end.html', {'msg': msg, 'risk': minRiskExpPortfolioReturn2, 'return': maxExpPortfolioReturn2, 'allocation': allocation, 'fundleft': fundLeft })

def abstract(request):
    return render(request, 'abstract.html')
