import os, json
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import metrics
import math
import dataframe_image as dfi


# Fetch Comapny fundamental data
def fetchFundamentalData():
    all500Tickers = dict()
    path_to_json_files = '../resources/data/'
    # get all JSON file names as a list
    json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

    for json_file_name in json_file_names:
        with open(os.path.join(path_to_json_files, json_file_name)) as json_file:
            json_text = json.load(json_file)
            all500Tickers.update(json_text)
    return all500Tickers

#Create dataframe with fundamental dict
def createDataframeWithFundamentalDict(fTickerDict):
    fIndex = [k for k, v in fTickerDict.items()]
    fColSet = set()
    for k, v in fTickerDict.items():
        for k1, v1 in v.items():
            fColSet.add(k1)
    fColList = list(fColSet)

    fDframe = pd.DataFrame(index=fIndex, columns=fColList)
    for k, v in fTickerDict.items():
        for index in fDframe.index:
            if index == k:
                for feature, itsValue in v.items():
                    fDframe.loc[index, feature] = itsValue
    return fDframe

# Fetch Pricing data
def fetchPricingData():
    priceData = pd.read_csv('../resources/data/all_500.csv')
    priceData = priceData.set_index('Date')
    priceData.to_csv("./downloads/pricing.csv")
    return priceData

def checkEmptyDataTicker(priceData):
    emptyColDropList = list()
    for ticker in priceData:
        if priceData[ticker].isna().sum() > 0:
            emptyColDropList.append(ticker)

    return emptyColDropList

def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows-1, Columns])
    for j in range(Columns):        # j: Assets
        for i in range(Rows-1):     # i: Daily Prices
            StockReturn[i,j]=((StockPrice.iloc[i+1, j]-StockPrice.iloc[i,j])/StockPrice.iloc[i,j])* 100
    np.savetxt("./downloads/dailyreturn.csv", StockReturn, delimiter=",")
    return StockReturn

def convertMillionToBillion(val):
    newVal=0.0
    if "M" in val:
        newVal=val.replace("M","")
        newVal=float(newVal)
        newVal*=0.001
    elif "B" in val:
        newVal=val.replace("B","")
        newVal= float(newVal)
    elif "T" in val:
        newVal=val.replace("T","")
        newVal= float(newVal)
        newVal*=1000
    elif 'k' in val:
        newVal=val.replace("k","")
        newVal=float(newVal)
    else:
        newVal=float(val)
    return newVal

def convertPercentToRatio(val):
    newVal = 0.0
    if '%' in val:
        newVal=val.replace("%","")
        newVal=float(newVal)
        newVal*=0.01
    elif np.empty(val):
        newVal=float(np.nan)
    elif 'k' in val:
        newVal=val.replace("k","")
        newVal=float(newVal)
    else:
        newVal=float(val)
    return newVal

def removeRandomChar(val):
    newVal = 0.0
    if 'k' in val:
        newVal=val.replace("k","")
        newVal=float(newVal)
    else:
        newVal=float(val.replace(",", ""))
    return newVal

def getLogReturnOnAsset(dframe, pFolioDict, portoflioName):
    logReturn = pd.DataFrame(index=dframe.columns, columns=['LogReturn'])
    for asset in dframe.columns:
            lastScaler = dframe[asset].iloc[-1]
            firstScaler = dframe[asset].iloc[0]
            changeLog = lastScaler - firstScaler
            logReturn.loc[asset,'LogReturn'] = changeLog
            pFolioDict[portoflioName]=logReturn
    return pFolioDict[portoflioName]

def fillMissingValue(portfolio):
    # Define a subset of the dataset
    portfolio = portfolio.copy()

    # Define scaler to set values between 0 and 1

    scaler = MinMaxScaler(feature_range=(0, 1))
    portfolio = pd.DataFrame(scaler.fit_transform(portfolio), index=portfolio.index, columns=portfolio.columns)
    # portfolio = pd.DataFrame(portfolio, index=portfolio.index, columns = portfolio.columns)

    # Define KNN imputer and fill missing values
    knnImputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    portfolio = pd.DataFrame(knnImputer.fit_transform(portfolio), index=portfolio.index, columns=portfolio.columns)
    return portfolio

def featureEngineering(fDframe, meanReturns, portfolio):
    # Risk Free Return
    numrtr = 1 + 3.962
    dnmrtr = 1 + 2.97
    div = numrtr / dnmrtr
    rf = div - 1
    # Lets consider price data return to be market return here; so to calculate CAPM we need market return,
    # 'in our scenario, we are going to take initial return calculated on pricing data as market return.
    weight = np.random.random(len(fDframe))
    weight /= weight.sum()
    weight = np.array(weight)

    expectedPortfolioReturn = meanReturns.T @ weight
    marketReturn = 251 * np.array(expectedPortfolioReturn)
    mrSubrf = marketReturn - rf
    mrSubrf[0]

    # Temporary dataframe to calculate CAPM for portfoliio1
    tempportfolio = pd.DataFrame(index=portfolio.index, columns=['Beta_(5Y_Monthly)', 'mrSubrf', 'Cost_of_Equity'])
    tempportfolio['Beta_(5Y_Monthly)'] = portfolio['Beta_(5Y_Monthly)']
    tempportfolio['mrSubrf'] = tempportfolio['Beta_(5Y_Monthly)'] * mrSubrf[0]
    tempportfolio['Cost_of_Equity'] = tempportfolio['mrSubrf'] + rf
    portfolio['Cost_of_Equity'] = tempportfolio['Cost_of_Equity']

    return portfolio

def firstCombinedTren(pFolioLog):
    import matplotlib.pyplot as plt0
    for k in pFolioLog:
        plt0.plot(pFolioLog[k].diff().mean(axis=1).cumsum(), label=k)
        plt0.xlabel('Date')
        plt0.ylabel('Cumulative Returns')
        plt0.title(f'Returns of portfolios over Time Period of Study')
        plt0.legend()
        plt0.savefig("./static/images/firstTrend.jpg")
    plt0.clf()

def downoadSNSPlot(portfolio, fileName):
    import matplotlib.pyplot as plt1
    import seaborn as sns
    sns.set()
    plt1.figure(figsize=(30, 15))
    sns.heatmap(portfolio.corr(), annot=True, cmap=plt1.cm.Accent)
    sns.barplot
    plt1.savefig(f"./static/images/{fileName}.jpg")
    plt1.clf()

def trainTestSplitAndFeatureRanking(portfolio, pNo):
    X = portfolio.drop(labels=['LogReturn'], axis=1)
    Y = portfolio['LogReturn']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=0)
    return X_train, X_test, y_train, y_test

# Display Matrices
def displayMetrics(y_te, y_pred, modelname, model, test_X, test_Y, X_tr, y_tr, ann=False):
    frameDict = dict()
    meanAbsError = metrics.mean_absolute_error(y_te, y_pred)
    meanSqrError = metrics.mean_squared_error(y_te, y_pred)
    rootMeanSqrError = math.sqrt(meanSqrError)
    rSqrTest = metrics.r2_score(y_te, y_pred)
    # rSqrTrain = metrics.r2_score(y_tr, y_pred)
    print("Mean Abs Error: {:.5f}".format(meanAbsError))
    print("Mean Sqr Error: {:.5f}".format(meanSqrError))
    print("Root Mean Sqr Error: {:.5f}".format(rootMeanSqrError))
    print("R square Test: {:.5f}".format(rSqrTest))
    # print("R square Trains: {:.5f}".format(rSqrTrain))
    frameDict['Model Name']=modelname
    frameDict['MeanAbsError']=meanAbsError
    frameDict['MeanSqrError']=meanSqrError
    frameDict['RootMeanSqrError']=rootMeanSqrError
    frameDict['R2SqrError']=rSqrTest
    if ann==False:
        crossValScore =cross_val_score(model, X_tr, y_tr, scoring = 'r2',cv=5).mean()
        print(f"{modelname} Cross val score: ", crossValScore)
        print(f"{modelname} score: ", model.score(test_X, test_Y) * 100)
        frameDict['CrossValScore']=crossValScore
    else:
        frameDict['CrossValScore']=0.0

    return frameDict

def displayMetricsForANN(y_te, y_pred, modelname, model, test_X, test_Y, X_tr, y_tr):
    frameDict= dict()
    meanAbsError = metrics.mean_absolute_error(y_te, y_pred)
    meanSqrError = metrics.mean_squared_error(y_te, y_pred)
    rootMeanSqrError = math.sqrt(meanSqrError)
    rSqrTest = metrics.r2_score(y_te, y_pred)
    # rSqrTrain = metrics.r2_score(y_tr, y_pred)
    print("Mean Abs Error: {:.5f}".format(meanAbsError))
    print("Mean Sqr Error: {:.5f}".format(meanSqrError))
    print("Root Mean Sqr Error: {:.5f}".format(rootMeanSqrError))
    print("R square Test: {:.5f}".format(rSqrTest))
    # print("R square Trains: {:.5f}".format(rSqrTrain))
    frameDict['Model Name']=modelname
    frameDict['MeanAbsError']=meanAbsError
    frameDict['MeanSqrError']=meanSqrError
    frameDict['RootMeanSqrError']=rootMeanSqrError
    frameDict['R2SqrError']=rSqrTest
    frameDict['CrossValScore']=0.0
    return frameDict

def buildAndTestModel(model, X_tr, y_tr, X_te, y_te, modelName):
    try:
        model.fit(X_tr, y_tr)
    except:
        try:
            model.fit(X_tr, y_tr.reshape(-1, 1))
        except:
            model.fit(X_tr, y_tr)

    # Test the model
    y_pred_test = model.predict(X_te)
    y_pred_train = model.predict(X_te)
    print(f"Test metrices from {modelName} are: \n")
    frameDict = displayMetrics(y_te, y_pred_test, modelName, model, X_te, y_te, X_tr, y_tr)
    return y_pred_test, y_te, frameDict

def linearRegression(modelEvalPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt2
    modelName =  f"Linear Regression Portfolio {pNo}"
    linearPort = LinearRegression()
    y_pred_p1_lin, y_test_p1_lin, frameDict = buildAndTestModel(linearPort, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    linearPortDict = {'y_pred_linear': pd.DataFrame(y_pred_p1_lin), 'y_test_linear': pd.DataFrame(y_test_p1_lin)}
    for k in linearPortDict:
        plt2.plot(linearPortDict[k].diff().mean(axis=1).cumsum(), label=k)
        plt2.title(f'Cumulative study of Actual & Predicted')
        plt2.legend()
    fileName = f"./static/images/linear_{pNo}.jpg"
    plt2.savefig(fileName)
    plt2.clf()

def decisionTree(modelEvalPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt3
    modelName = f"Decision Tree Portfolio {pNo}"
    dtPortfolio = DecisionTreeRegressor(max_depth=5)
    y_pred_p1_dt, y_test_p1_dt, frameDict = buildAndTestModel(dtPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    dtPort1Dict = {'y_pred_dt': pd.DataFrame(y_pred_p1_dt), 'y_test_dt': pd.DataFrame(y_test_p1_dt)}
    for k in dtPort1Dict:
        plt3.plot(dtPort1Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt3.title(f'Cumulative study of Actual & Predicted')
        plt3.legend()
    fileName = f"./static/images/dt_{pNo}.jpg"
    plt3.savefig(fileName)
    plt3.clf()
    return frameDict

def supportVectorMachine(modelEvalPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt4
    svmPortfolio = SVR()
    modelName = f"SVM Portfolio {pNo}"
    y_pred_p2_svm, y_test_p2_svm, frameDict = buildAndTestModel(svmPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    svmPort2Dict = {'y_pred_svm': pd.DataFrame(y_pred_p2_svm), 'y_test_svm': pd.DataFrame(y_test_p2_svm)}
    for k in svmPort2Dict:
        plt4.plot(svmPort2Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt4.title(f'Cumulative study of Actual & Predicted')
        plt4.legend()
    fileName = f"./static/images/svm_{pNo}.jpg"
    plt4.savefig(fileName)
    plt4.clf()
    return frameDict

def lassoRegression(modelEvalPortfolio, alpha, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt5
    lassoPortfolio = linear_model.Lasso(alpha=alpha,
                                         precompute=True,
                                         positive=True,
                                         selection='random',
                                         random_state=42)
    modelName = f"Lasso Portfolio {pNo}"
    y_pred_p1_lasso, y_test_p1_lasso, frameDict = buildAndTestModel(lassoPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    lassoPort1Dict = {'y_pred_lasso': pd.DataFrame(y_pred_p1_lasso), 'y_test_lasso': pd.DataFrame(y_test_p1_lasso)}
    for k in lassoPort1Dict:
        plt5.plot(lassoPort1Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt5.title(f'Cumulative study of Actual & Predicted')
        plt5.legend()
    fileName = f"./static/images/lasso_{pNo}.jpg"
    plt5.savefig(fileName)
    plt5.clf()
    return frameDict

def randomForestRegressor(modelEvalPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt6
    randforPortfolio = RandomForestRegressor(max_depth=100)
    modelName = f"Random Forest Portfolio {pNo}"
    y_pred_p1_randforestreg, y_test_p1_randforestreg, frameDict = buildAndTestModel(randforPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    ranforPort1Dict = {'y_pred_p1_randforestreg': pd.DataFrame(y_pred_p1_randforestreg),
                       'y_test_p1_randforestreg': pd.DataFrame(y_test_p1_randforestreg)}
    for k in ranforPort1Dict:
        plt6.plot(ranforPort1Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt6.title(f'Cumulative study of Actual & Predicted')
        plt6.legend()
    fileName = f"./static/images/randforest_{pNo}.jpg"
    plt6.savefig(fileName)
    plt6.clf()
    return frameDict

def ridgeRegression(modelEvalPortfolio, tol, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt7
    ridgePortfolio = Ridge(alpha=10, solver='cholesky', tol=tol, random_state=42)
    modelName = f"Ridge Portfolio {pNo}"
    y_pred_p1_ridge, y_test_p1_ridge, frameDict = buildAndTestModel(ridgePortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    ridgePort1Dict = {'y_pred_p1_ridge': pd.DataFrame(y_pred_p1_ridge),
                      'y_test_p1_ridge': pd.DataFrame(y_test_p1_ridge)}
    for k in ridgePort1Dict:
        plt7.plot(ridgePort1Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt7.title(f'Cumulative study of Actual & Predicted')
        plt7.legend()
    fileName = f"./static/images/ridge_{pNo}.jpg"
    plt7.savefig(fileName)
    plt7.clf()
    return frameDict

def elasticNet(modelEvalPortfolio, alpha, l1_ratio, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt8
    elasticNetPortfolio = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, selection='random', random_state=42)
    modelName = f"Elastic Net Portfolio {pNo}"
    y_pred_p1_elastic, y_test_p1_elastic, frameDict = buildAndTestModel(elasticNetPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    elasticPort1Dict = {'y_pred_p1_elastic': pd.DataFrame(y_pred_p1_elastic),
                        'y_test_p1_elastic': pd.DataFrame(y_test_p1_elastic)}
    for k in elasticPort1Dict:
        plt8.plot(elasticPort1Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt8.title(f'Cumulative study of Actual & Predicted')
        plt8.legend()
    fileName = f"./static/images/elastic_{pNo}.jpg"
    plt8.savefig(fileName)
    plt8.clf()
    return frameDict

def sgdModel(modelEvalPortfolio, n_iter_no_change, x1_Tr, y1_Tr, x1_Te, y1_Te, pNo):
    import matplotlib.pyplot as plt9
    sgdPortfolio = SGDRegressor(n_iter_no_change=n_iter_no_change, penalty=None, eta0=0.001, max_iter=1000000000)
    modelName = f"SGD Portfolio {pNo}"
    y_pred_p1_sgd, y_test_p1_sgd, frameDict = buildAndTestModel(sgdPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te, modelName)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    sgdPort1Dict = {'y_pred_p1_sgd': pd.DataFrame(y_pred_p1_sgd), 'y_test_p1_sgd': pd.DataFrame(y_test_p1_sgd)}
    for k in sgdPort1Dict:
        plt9.plot(sgdPort1Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt9.title(f'Cumulative study of Actual & Predicted')
        plt9.legend()
    fileName = f"./static/images/sgd_{pNo}.jpg"
    plt9.savefig(fileName)
    plt9.clf()
    return frameDict

def ann1(modelEvalPortfolio, x1_Tr, y1_Tr, x1_Te, y1_Te):
    import matplotlib.pyplot as plt10
    modelP = Sequential()
    # x1_Tr, y1_Tr
    modelP.add(Dense(x1_Tr.shape[1], activation='relu'))
    modelP.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.2))

    modelP.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))

    modelP.add(Dense(64, activation='relu'))

    modelP.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.1))
    modelP.add(Dense(1))

    modelP.compile(optimizer=Adam(0.00001), loss='mse')

    r = modelP.fit(x1_Tr, y1_Tr,
                    validation_data=(x1_Te, y1_Te),
                    batch_size=1,
                    epochs=100)
    
    testPredANN1 = modelP.predict(x1_Te)
    trainPredANN1 = modelP.predict(x1_Tr)
    frameDict = displayMetrics(y1_Te, testPredANN1, "Artificial Neural Network 1", modelP, x1_Te, y1_Te, x1_Tr, y1_Tr, ann=True)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    annPort1Dict = {'test_pred': pd.DataFrame(testPredANN1), 'y1_Te': pd.DataFrame(y1_Te)}
    for k in annPort1Dict:
        plt10.plot(annPort1Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt10.title(f'Cumulative study of Actual & Predicted')
        plt10.legend()

    fileName = f"./static/images/ann_1.jpg"
    plt10.savefig(fileName)
    plt10.clf()
    return frameDict

def ann2(modelEvalPortfolio, x2_Tr, y2_Tr, x2_Te, y2_Te):
    import matplotlib.pyplot as plt11
    modelP2 = Sequential()

    modelP2.add(Dense(x2_Tr.shape[1], activation='relu'))
    modelP2.add(Dense(32, activation='relu'))
    modelP2.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))

    modelP2.add(Dense(128, activation='relu'))

    modelP2.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.1))
    modelP2.add(Dense(1))

    modelP2.compile(optimizer=Adam(0.00001), loss='mse')

    r = modelP2.fit(x2_Tr, y2_Tr,
                    validation_data=(x2_Te, y2_Te),
                    batch_size=1,
                    epochs=100)
    testPredANN2 = modelP2.predict(x2_Te)
    trainPredANN2 = modelP2.predict(x2_Tr)
    frameDict = displayMetrics(y2_Te, testPredANN2, "Artificial Neural Network", modelP2, x2_Te, y2_Te, x2_Tr, y2_Tr,
                               ann=True)
    # modelEvalPortfolio = modelEvalPortfolio._append(frameDict, ignore_index=True)

    annPort2Dict = {'test_pred': pd.DataFrame(testPredANN2), 'train_pred': pd.DataFrame(trainPredANN2)}
    for k in annPort2Dict:
        plt11.plot(annPort2Dict[k].diff().mean(axis=1).cumsum(), label=k)
        plt11.title(f'Cumulative study of Actual & Predicted')
        plt11.legend()
    fileName = f"./static/images/ann_2.jpg"
    plt11.savefig(fileName)
    plt11.savefig(fileName)
    plt11.clf()
    return frameDict

def dataframToImage(df,fileName):
    import matplotlib.pyplot as plt12
    df.dfi.export(fileName)
    plt12.clf()

##Return and Risk Logic
# function obtains maximal return portfolio using linear programming
def MaximizeReturns(MeanReturns, PortfolioSize):
    # dependencies
    from scipy.optimize import linprog
    import numpy as np

    c = (np.multiply(-1, MeanReturns))
    A = np.ones([PortfolioSize, 1]).T
    b = [1]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='simplex')

    return res


# function obtains minimal risk portfolio
# dependencies
from scipy import optimize
def MinimizeRisk(CovarReturns, PortfolioSize):
    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T) - b
        return constraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(f, x0=xinit, args=(CovarReturns), bounds=bnds, \
                            constraints=cons, tol=10 ** -3)
    return opt


# function obtains Minimal risk and Maximum return portfolios
def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        AEq = np.ones(x.shape)
        bEq = 1
        EqconstraintVal = np.matmul(AEq, x.T) - bEq
        return EqconstraintVal

    def constraintIneq(x, MeanReturns, R):
        AIneq = np.array(MeanReturns)
        bIneq = R
        IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
        return IneqconstraintVal

    xinit = np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq},
            {'type': 'ineq', 'fun': constraintIneq, 'args': (MeanReturns, R)})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = optimize.minimize(f, args=(CovarReturns), method='trust-constr', \
                            x0=xinit, bounds=bnds, constraints=cons, tol=10 ** -3)
    return opt


# function computes asset returns
def PortfolioReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100
    return StockReturn

def priceDivision(portfolio, budget):
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage

    mu = mean_historical_return(portfolio)
    S = CovarianceShrinkage(portfolio).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(portfolio)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=budget)

    allocation, leftover = da.greedy_portfolio()
    fundLeft = "${:.2f}".format(leftover)
    return allocation, fundLeft

def returnCalculator(portfolio, priceData, budget):
    p2Cal = pd.DataFrame()
    for i in portfolio.index:
        p2Cal.loc[:, i] = priceData.loc[:, i]
    r1, c1 = p2Cal.shape
    # extract asset labels
    assetLabels2 = p2Cal.columns[1:c1 + 1].tolist()
    print(assetLabels2)
    arStockPrices2 = p2Cal

    # compute asset returns
    # arStockPrices = np.asarray(StockData)
    [Rows2, Cols2] = arStockPrices2.shape
    print(Rows2, Cols2)
    arReturns2 = np.array(StockReturnsComputing(arStockPrices2, Rows2, Cols2))

    # compute mean returns and variance covariance matrix of returns
    meanReturns2 = np.mean(arReturns2, axis=0)
    covReturns2 = np.cov(arReturns2, rowvar=False)

    # set precision for printing results
    np.set_printoptions(precision=3, suppress=True)

    result2 = MaximizeReturns(meanReturns2, Cols2)
    maxReturnWeights2 = result2.x
    maxExpPortfolioReturn2 = np.matmul(meanReturns2.T, maxReturnWeights2)
    maxExpPortfolioReturn2 = round(maxExpPortfolioReturn2, 4) * 100

    result21 = MinimizeRisk(covReturns2, Cols2)
    minRiskWeights2 = result21.x
    minRiskExpPortfolioReturn2 = np.matmul(meanReturns2.T, minRiskWeights2)
    minRiskExpPortfolioReturn2 = round(minRiskExpPortfolioReturn2, 4) * 100

    allocation, fundLeft = priceDivision(p2Cal, budget)
    print(allocation)
    print(fundLeft)
    return maxExpPortfolioReturn2, minRiskExpPortfolioReturn2, allocation, fundLeft


# our result page view
def getUserInputs(request):
    usersInput = list()
    communicationServices = bool(request.GET.get('CommunicationServices', False))
    consumerDefensive = bool(request.GET.get('ConsumerDefensive', False))
    consumerCyclical = bool(request.GET.get('ConsumerCyclical', False))
    financialServices = bool(request.GET.get('FinancialServices', False))
    industrialGoods = bool(request.GET.get('IndustrialGoods', False))
    basicMaterials = bool(request.GET.get('BasicMaterials', False))
    technology = bool(request.GET.get('Technology', False))
    healthcare = bool(request.GET.get('Healthcare', False))
    industrials = bool(request.GET.get('Industrials', False))
    realEstate = bool(request.GET.get('RealEstate', False))
    utilities = bool(request.GET.get('Utilities', False))
    energy = bool(request.GET.get('Energy', False))
    allIndsutry = bool(request.GET.get('All', False))

    sectorDict = {'Communication Services': communicationServices,
                  'Consumer Defensive': consumerDefensive,
                  'Consumer Cyclical':consumerCyclical,
                  'Financial Services':financialServices,
                  'Industrial Goods':industrialGoods,
                  'Basic Materials':basicMaterials,
                  'Technology':technology,
                  'Healthcare':healthcare,
                  'Industrials':industrials,
                  'Real Estate':realEstate,
                  'Utilities':utilities,
                  'Energy':energy}

    budget = float(request.GET.get('budget', 10))

    budgetDict = {'budget':budget}


    if allIndsutry == False:
        usersInput = [sectorDict, budgetDict]
    else:
        sectorDict = {'Communication Services': True,
                      'Consumer Defensive': True,
                      'Consumer Cyclical': True,
                      'Financial Services': True,
                      'Industrial Goods': True,
                      'Basic Materials': True,
                      'Technology': True,
                      'Healthcare': True,
                      'Industrials': True,
                      'Real Estate': True,
                      'Utilities': True,
                      'Energy': True}
        usersInput = [sectorDict, budgetDict]

    usersInput = [{'Confirm Selection': True}, budgetDict]
    return usersInput

