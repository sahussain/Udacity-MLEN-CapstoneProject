# Machine Learning Engineer Nanodegree
## Capstone Project
Ashraf Hussain 
22 June, 2020

Machine Learning Engineer Nanodegree: Forecasting COVID-19 Cases – A Time Series Forecasting Model

## I. Definition


### Project Overview

On 31 December, 2019, the World Health Organization (WHO) was informed of an outbreak of “pneumonia of unknown cause” detected in Wuhan City, Hubei Province, China. Initially identified as coronavirus disease 2019, it quickly came to be known widely as COVID-19 and has resulted in an ongoing global pandemic. As of 20 June, 2020, more than 8.74 million cases have been reported across 188 countries and territories, resulting in more than 462,000 deaths. More than 4.31 million people have recovered.[^1]

In response to this ongoing public health emergency, Johns Hopkins
University (JHU), a private research university in Maryland, USA,
developed an interactive web-based dashboard hosted by their Center for
Systems Science and Engineering (CSSE). The dashboard visualizes and
tracks reported cases in real-time, illustrating the location and number
of confirmed COVID-19 cases, deaths and recoveries for all affected
countries. It is used by researchers, public health authorities, news
agencies and the general public. All the data collected and displayed is
made freely available in a [GitHub repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data).


### Problem Statement
This project seeks to forecast number of people infected and number of
caused by COVID-19 for a time duration of 14-days based on
historical data from JHU. I will be using Amazon SageMaker DeepAR
forecasting algorithm, a supervised learning algorithm for forecasting
scalar (one-dimensional) time series using recurrent neural networks (RNN)
to produce both point and probabilistic forecasts[^2].
DeepAR is an underutilized approach in this area.[^3] The dataset contains
hundreds of related time series, and DeepAR outperforms classical
forecasting methods including but not limited to autoregressive integrated
moving average (ARIMA), exponential smoothing (ETS), Time Series
Forecasting with Linear Learner for this type of applications.

At first, I was going to use [DeepAR by AWS]([https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)) and compare it to  AR (autoregressive model). However after exploring the data, I discovered that this dataset cannot be used for any Time Series Forecasting Model(TSFMs). Epidemic curves (epi curve) do not follow a standard time series requirement however they do follow Logistic & Gaussian functions that are defined as follows:

Epi curve of total number of cases follows Logistic Function is defined by:
>f(x) = capacity / (1 + e^-k*(x - midpoint) )[^4]

Epi curve of new of cases follows Gaussian Function is defined by:
>f(x) = a * e^(-0.5 * ((x-μ)/σ)**2)[^4]

For this reason, in the dataset model I propose total cases will fit to to Logistic function and new cases to Gaussian functions.

### Metrics
The error represents random variations in the data that follow a specific probability distribution (usually Gaussian). The objective of curve fitting is to find the optimal combination of parameters that minimize the error. Here we are dealing with time series, therefore the independent variable is time. In mathematical terms[^4]
>f(error) = f(time) + error


## II. Analysis

### Data Exploration
The datasets are accessed from files provided by the JHU GitHub
repository [time_series_covid19_confirmed_US.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv)

The file have the same columns:
* UID - UID = 840 (country code3) + 000XX (state FIPS code). Ranging from
8400001 to 84000056.
* iso2- Officially assigned country code identifiers 2 Chr (US, CA, ...)
* iso3 - Officially assigned country code identifiers 3 Chr.(USA, CAN,
...)
* code3- country code USA = 840
* FIPS -Federal Information Processing Standards code that uniquely
identifies counties within the USA.
* admin2 - County name. US only.
* Province_State - The name of the State within the USA.
* Country_Region - The name of the Country (US).
* Combined_Key - Province_State + Country_Region
* Population - Population
* Number of cases are is columns where each column is a day


### Exploratory Visualization
The plot below shows how the COVID-19 cases increase by city. When looking at the graph of 15 cities I noticed the first problem. I cannot use DeepAR on this data because DeepAR needs at least 300 observations available across all training time series. The current dataset has observations for the last 151 days only.

>We recommend training a DeepAR model on as many time series as are available. Although a DeepAR model trained on a single time series might work well, standard forecasting algorithms, such as ARIMA or ETS, might provide more accurate results. The DeepAR algorithm starts to outperform the standard methods when your dataset contains hundreds of related time series. Currently, DeepAR requires that the total number of observations available across all training time series is at least 300. 
>Source: https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html

![enter image description here](/Images/Capture2.JPG)

Lets look at new vs total number of cases for some cities:
![enter image description here](/Images/Capture3.JPG)
![enter image description here](/Images/Capture4.JPG)





### Algorithms and Techniques
To use DeepAR it needs to meet the following criteria [^5]:
1. Except for when splitting your dataset for training and testing, always provide the entire time series for training, testing, and when calling the model for inference. Regardless of how you set `context_length`, don't break up the time series or provide only a part of it. The model uses data points further back than the value set in `context_length` for the lagged values feature.[^5]
2. When tuning a DeepAR model, you can split the dataset to create a training dataset and a test dataset. In a typical evaluation, you would test the model on the same time series used for training, but on the future  `prediction_length`  time points that follow immediately after the last time point visible during training. You can create training and test datasets that satisfy this criteria by using the entire dataset (the full length of all time series that are available) as a test set and removing the last  `prediction_length`  points from each time series for training. During training, the model doesn't see the target values for time points on which it is evaluated during testing. During testing, the algorithm withholds the last  `prediction_length`  points of each time series in the test set and generates a prediction. Then it compares the forecast with the withheld values. You can create more complex evaluations by repeating time series multiple times in the test set, but cutting them at different endpoints. With this approach, accuracy metrics are averaged over multiple forecasts from different time points. For more information, see  [Tune a DeepAR Model](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar-tuning.html).[^5]

3. Avoid using very large values (>400) for the  `prediction_length`  because it makes the model slow and less accurate. If you want to forecast further into the future, consider aggregating your data at a higher frequency. For example, use  `5min`  instead of  `1min`.[^5]
    
4. Because lags are used, a model can look further back in the time series than the value specified for  `context_length`. Therefore, you don't need to set this parameter to a large value. We recommend starting with the value that you used for  `prediction_length`.[^5]
    
5. We recommend training a DeepAR model on as many time series as are available. Although a DeepAR model trained on a single time series might work well, standard forecasting algorithms, such as ARIMA or ETS, might provide more accurate results. The DeepAR algorithm starts to outperform the standard methods when your dataset contains hundreds of related time series. Currently, DeepAR requires that the total number of observations available across all training time series is at least 300.[^5]

Based on the above requirements we can not use DeepAR. I then looked into the following Then I looked into the following models to see if we can use them:

 ## Researching  `Time Series Forecasting Model(TSFMs)`

***AR (autoregressive model):***
The autoregression (AR) method models the next step in the sequence as a linear function of the observations at prior time steps.
The notation for the model involves specifying the order of the model p as a parameter to the AR function, e.g. AR(p). For example, AR(1) is a first-order autoregression model.
The method is suitable for univariate time series without trend and seasonal components.[^1]

***MA (moving-average model):***
The moving average (MA) method models the next step in the sequence as a linear function of the residual errors from a mean process at prior time steps.
A moving average model is different from calculating the moving average of the time series.
The notation for the model involves specifying the order of the model q as a parameter to the MA function, e.g. MA(q). For example, MA(1) is a first-order moving average model.
The method is suitable for univariate time series without trend and seasonal components.[^1]

***ARMA (autoregressive-moving-average model):***
The Autoregressive Moving Average (ARMA) method models the next step in the sequence as a linear function of the observations and resiudal errors at prior time steps.
It combines both Autoregression (AR) and Moving Average (MA) models.
The notation for the model involves specifying the order for the AR(p) and MA(q) models as parameters to an ARMA function, e.g. ARMA(p, q). An ARIMA model can be used to develop AR or MA models.
The method is suitable for univariate time series without trend and seasonal components.[^6]

***ARIMA (autoregressive integrated moving average model):***
The Autoregressive Integrated Moving Average (ARIMA) method models the next step in the sequence as a linear function of the differenced observations and residual errors at prior time steps.
It combines both Autoregression (AR) and Moving Average (MA) models as well as a differencing pre-processing step of the sequence to make the sequence stationary, called integration (I).
The notation for the model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function, e.g. ARIMA(p, d, q). An ARIMA model can also be used to develop AR, MA, and ARMA models.
The method is suitable for univariate time series with trend and without seasonal components.[^6]

***SARIMA (seasonal autoregressive integrated moving average model):***
The  [Seasonal Autoregressive Integrated Moving Average (SARIMA)](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/)  method models the next step in the sequence as a linear function of the differenced observations, errors, differenced seasonal observations, and seasonal errors at prior time steps.
It combines the ARIMA model with the ability to perform the same autoregression, differencing, and moving average modeling at the seasonal level.
The notation for the model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function and AR(P), I(D), MA(Q) and m parameters at the seasonal level, e.g. SARIMA(p, d, q)(P, D, Q)m where “m” is the number of time steps in each season (the seasonal period). A SARIMA model can be used to develop AR, MA, ARMA and ARIMA models.
The method is suitable for univariate time series with trend and/or seasonal components.[^6]

***SARIMAX (seasonal autoregressive integrated moving average model with exogenous variables):***
The Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors ([SARIMAX](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/)) is an extension of the SARIMA model that also includes the modeling of exogenous variables.
Exogenous variables are also called covariates and can be thought of as parallel input sequences that have observations at the same time steps as the original series. The primary series may be referred to as endogenous data to contrast it from the exogenous sequence(s). The observations for exogenous variables are included in the model directly at each time step and are not modeled in the same way as the primary endogenous sequence (e.g. as an AR, MA, etc. process).
The SARIMAX method can also be used to model the subsumed models with exogenous variables, such as ARX, MAX, ARMAX, and ARIMAX.
The method is suitable for univariate time series with trend and/or seasonal components and exogenous variables.[^6]

***VARMA (vector autoregressive moving average model):***
The Vector Autoregression Moving-Average (VARMA) method models the next step in each time series using an ARMA model. It is the generalization of ARMA to multiple parallel time series, e.g. multivariate time series.
The notation for the model involves specifying the order for the AR(p) and MA(q) models as parameters to a VARMA function, e.g. VARMA(p, q). A VARMA model can also be used to develop VAR or VMA models.
The method is suitable for multivariate time series without trend and seasonal components.[^6]

-----

***Vector Autoregression (VAR)***
The Vector Autoregression (VAR) method models the next step in each time series using an AR model. It is the generalization of AR to multiple parallel time series, e.g. multivariate time series.
The notation for the model involves specifying the order for the AR(p) model as parameters to a VAR function, e.g. VAR(p).
The method is suitable for multivariate time series without trend and seasonal components.[^6]

***Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)***
The Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX) is an extension of the VARMA model that also includes the modeling of exogenous variables. It is a multivariate version of the ARMAX method.
Exogenous variables are also called covariates and can be thought of as parallel input sequences that have observations at the same time steps as the original series. The primary series(es) are referred to as endogenous data to contrast it from the exogenous sequence(s). The observations for exogenous variables are included in the model directly at each time step and are not modeled in the same way as the primary endogenous sequence (e.g. as an AR, MA, etc. process).
The VARMAX method can also be used to model the subsumed models with exogenous variables, such as VARX and VMAX.
The method is suitable for multivariate time series without trend and seasonal components with exogenous variables.[^6]

***Simple Exponential Smoothing (SES)***
The Simple Exponential Smoothing (SES) method models the next time step as an exponentially weighted linear function of observations at prior time steps.
The method is suitable for univariate time series without trend and seasonal components.[^6]


***Holt Winter’s Exponential Smoothing (HWES)***

The  [Holt Winter’s Exponential Smoothing](https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/)  (HWES) also called the Triple Exponential Smoothing method models the next time step as an exponentially weighted linear function of observations at prior time steps, taking trends and seasonality into account.
The method is suitable for univariate time series with trend and/or seasonal components.[^6]

***ARCH (autoregressive conditional heteroscedasticity model)***
Autoregressive conditional heteroskedasticity (ARCH) is a time-series statistical model used to analyze effects left unexplained by econometric models. In these models, the error term is the residual result left unexplained by the model. The assumption of econometric models is that the [variance](https://www.investopedia.com/terms/v/variance.asp) of this term will be uniform. This is known as "homoskedasticity." However, in some circumstances, this variance is not uniform, but "heteroskedastic."[^7]
***ARIMAX (autoregressive integrated moving average model with exogenous variables)***
A time series model using the Autoregressive Integrated Moving Average with exogenous variables (ARIMAX) function was developed to predict impacts from groundwater pumping on Silver Springs discharge in Ocala Florida. This effort was conducted to determine the effects of groundwater withdrawal using the statistical relationship between rainfall and spring discharge at Silver Springs. Other statistical models were developed in previous work by both Southwest Florida Water Management Districts and St Johns River Water Management District. However, there were several opportunities for improvement including using consistent data, model calibration period, and residual periods. Additionally previous statistical methods included Multiple Linear Regression and Line of Organic Correlation methods. These methods did not account for autocorrelation that present in many time series analysis. Through inter-district collaboration, data was made consistent and new methods were explored. The ARIMAX model was explored in this paper and is useful for prediction when autoregressive patterns are present in model residuals that bias modeled coefficients.[^8]
***GARCH (generalized autoregressive conditional heteroscedasticity model)***
A natural generalization of the ARCH (Autoregressive Conditional Heteroskedastic) process introduced in Engle (1982) to allow for past conditional variances in the current conditional variance equation is proposed. Stationarity conditions and autocorrelation structure for this new class of parametric models are derived. Maximum likelihood estimation and testing are also considered.[^9] 

To see if we can use TSFMs, the first thing we need to do is to check for Stationarity. A common assumption in many time series techniques is that the data are stationary.
A stationary process has the property that the mean, variance and autocorrelation structure do not change over time. Stationarity can be defined in precise mathematical terms, but for our purpose we mean a flat looking series, without trend, constant variance over time, a constant autocorrelation structure over time and no periodic fluctuations (seasonality).

For practical purposes, stationarity can usually be determined from a run sequence plot.

For a time-series to be Stationarity, the following do not change over time

mean(μ)
standard deviation(σ)
Autocorrelation structure (No seasonality)
for more info see Stationarity

There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used. It uses an autoregressive model and optimizes an information criterion across multiple different lag values.

The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.

Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure. Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure. We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).

-   p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
-   p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

source:  [How to Check if Time Series Data is Stationary with Python](https://machinelearningmastery.com/time-series-data-stationary-python/)

Want to know more about  [How to Check if Time Series Data is Stationary with Python](https://machinelearningmastery.com/time-series-data-stationary-python/)

> If we fit a stationary model to data, we assume our data are a realization of a stationary process. So our first step in an analysis should be to check whether there is any evidence of a trend or seasonal effects and, if there is, remove them.

— Page 122,  [Introductory Time Series with R](http://www.amazon.com/dp/0387886974?tag=inspiredalgor-20).

`The problem is that we can not remove any data`

so what can we do? Lets look at one state  `New York`

I used the following function to check for Stationarity:

```Python 
def getStationarity(df, States):
    #print(ts[0])
    df_return = pd.DataFrame() 
    arr_state = []
    arr_adf = []
    arr_p_value = []
    arr_usedlag = [] 
    arr_nobs = []
    arr_1 =[]
    arr_5 = []
    arr_10 = []
    icbest=[]
    res = []
    cit_stationary =[]
    
    threshold = 0.05
    print("### Testing Null Hypothesis ###")
    for aState in States:
        try:
            Cases=getCases(df, aState)
            #print(Cases[0])
            adfuller_results = sts.adfuller(Cases[0].total)
            #print(adfuller_results)
            arr_state.append(aState)
            arr_adf.append(round(adfuller_results[0],3))
            if round(adfuller_results[1],3)>threshold:
                res.append(0) #0 implies non-stationary
            else:
                res.append(1) #1 implies stationary
                cit_stationary.append(aState)
            arr_p_value.append(round(adfuller_results[1],3))
            arr_usedlag.append(round(adfuller_results[2],3))
            arr_nobs.append(round(adfuller_results[3],3))
            arr_1.append(round(adfuller_results[4]['1%'],3))
            arr_5.append(round(adfuller_results[4]['5%'],3))
            arr_10.append(round(adfuller_results[4]['10%'],3))
            icbest.append(round(adfuller_results[3],3))
            #print("\n\n")
        except:
            print("An occurred with city: ", aState)
    #print(arr_1)
    df_return['State'] = arr_state
    df_return['stationary'] = res
    df_return['adf'] = arr_adf
    df_return['p-value'] = arr_p_value
    df_return['usedlag'] = arr_usedlag
    df_return['nobs'] = arr_nobs
    df_return['significance_1'] = arr_1
    df_return['significance_5'] = arr_5
    df_return['significance_10'] = arr_10
    df_return['icbest'] = icbest
    
    print("Calculation Complete")
    print("adfuller results:")
    print(str(len(States)-len(cit_stationary)) + " p-value >  0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.")
    print(str(len(cit_stationary)) + " p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.")
    print("--------------------")
    print(str(len(States)) + " Total")
    
    #df_return.set_index('City')
    #return None
    return df_return, cit_stationary

stationarity_df = getStationarity(covid_df, state[0:])
```
Testing Null Hypothesis
Calculation Complete
adfuller results:
56 p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
2 p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
58 Total




I will be using [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html), which is a part of [SciPy](https://scipy.org/)
package. This will be fitting a pre-defined Gaussian/Logistic Function which is very commonly used in epidemiology. [^7]




### Benchmark
We are still in a crucial stage of the epidemic; there is no real benchmark. The model was able to predict future cases for some states very well. However, for others it was unable to predict at all. I used a random generator to pick sample cities to plot the following graphs:
**Delaware**
![enter image description here](/Images/Capture5.JPG)
In Delaware, the number of new cases seem to be following an epi curve very well. This indicates that the social distancing measures have been working well to reduce the number of new cases. There was a spike in the number of cases in the week of Memorial Day (May 25), however, they seemed to have recovered quickly falling into a standard epi curve pattern.

**North Dakota**
![enter image description here](/Images/Capture6.JPG)
New covid-19 cases in North Dakota seem to be following the same pattern as that of Delaware indicating that social distancing measures are proving to be effective. Their cases peaked around Memorial Day, similar to Delaware, hwoever they seem to have take slightly longer to come back to a normal epi curve.

**Maryland**
![enter image description here](/Images/Capture7.JPG)
Contrary to Delaware and North Dakota, new cases in Maryland peaked during the long weekend in April and seemed to stay within the mean of an epi curve. This irregular peaking may be attributed to people traveling between states for the long weekend and for other reasons.

## III. Methodology

### Data Preprocessing
The [time_series_covid19_confirmed_US.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv) 

**UID**|**iso2**|**iso3**|**code3**|**FIPS**|**Admin2**|**Province\_State**|**Country\_Region**|**Lat**|**Long\_**|**...**|**6/13/20**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
16|AS|ASM|16|60|NaN|American Samoa|US|-14.271|-170.132|...|0
316|GU|GUM|316|66|NaN|Guam|US|13.4443|144.7937|...|183
580|MP|MNP|580|69|NaN|Northern Mariana Islands|US|15.0979|145.6739|...|30
630|PR|PRI|630|72|NaN|Puerto Rico|US|18.2208|-66.5901|...|5690
850|VI|VIR|850|78|NaN|Virgin Islands|US|18.3358|-64.8963|...|72

The data set was imported into a pandas Dataframe. 

Data needed minimal data preprocessing because each Date was in one column and City and State were in other columns.
```python
csv_file = 'time_series_covid19_confirmed_US.csv'
covid_df = pd.read_csv(csv_file)
```

Then the data was modified to remove the following columns and each state was sum
```python
covid_df = covid_df.drop(['UID',
	                 'iso2',
	                 'iso3',
	                 'code3',
	                 'FIPS',
	                 'Admin2',
	                 'Country_Region',
	                 'Lat',
	                 'Long_',
	                 'Combined_Key'], axis=1).groupby("Province_State").sum().T
```

Province\_State|Alabama|Alaska|American Samoa|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Diamond Princess|...|Tennessee|Texas|Utah|Vermont|Virgin Islands|Virginia|Washington|West Virginia|Wisconsin|Wyoming
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
1/22/20|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|1|0|0|0
1/23/20|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|1|0|0|0
1/24/20|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|1|0|0|0
1/25/20|0|0|0|0|0|0|0|0|0|0|...|0|0|0|0|0|0|1|0|0|0
1/26/20|0|0|0|1|0|2|0|0|0|0|...|0|0|0|0|0|0|1|0|0|0
 | | | | | | | | | | | | | | | | | | | | | 
5 rows × 58 columns| | | | | | | | | | | | | | | | | | | | | 


The dataframe Date was converted to datetime Index
```python
## convert index to datetime
covid_df.index = pd.to_datetime(covid_df.index, infer_datetime_format=True)
```

Now we have a clean data set which will have Date as Index, and sum of cases for each State.
From here onwards, we can use the following function to get cumulative number of cases (total) and new cases (new) for a given State in the dataframe.
```Python
def getCases(df, aState):
    # create total cases column
    error = 0
    try:
        df = pd.DataFrame(index=df.index, data=df[aState].values, columns=["total"])
        #print(dtf.head())
        # create daily changes column
        df["new"] = df["total"] - df["total"].shift(1)
        # Handling Missing Values
        df["new"] = df["new"].fillna(method='bfill')
    except:
        print("No State " + aState + " found")
        error = 1
        df = pd.DataFrame() 
    return [df, error]
```
.|total|new
:-----:|:-----:|:-----:
count|153|153
mean|168552.3987|2539.137255
std|162319.6416|3209.162717
min|0|0
25%|0|0
50%|139875|1075
75%|345813|4073
max|388488|11434

### Implementation
Model:
[scipy.optimize.curve_fit model  from scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) provides a non-linear least squares to fit a function, f (such as Logistic or Gaussian functions) to a given dataframe. 
Forecast:
The forecasting function was provided by [`ts_utils.py`](https://github.com/mdipietro09/DataScience_ArtificialIntelligence_Utils/blob/master/time_series/ts_utils.py)
that apply the two models (total cases and daily increase) to a new independent variable: the time steps from today till N. It forecast 30 days ahead from today[^8]


### Refinement
The model and the forecast work well when applied to certain States. However, there are a few States that do not work because they are on a different epi curve, for example
 
 **Michigan**
![enter image description here](/Images/Capture9.JPG)
The model could not fit total cases to a Logistic Function

**Virgin Islands**
![enter image description here](/Images/Capture11.JPG)
The model could not fit new cases to a Gaussian Function

A better package such as  `earlyR`  and  `EpiEstim`  that are part of R work better when applied to an epidemic dataframe. 
There are two well-researched articles written by [Tim Churches](https://theconversation.com/profiles/timothy-churches-1003068) that talk about using R to predict COVID-19 cases. 
>Tim Churches is a Senior Research Fellow at the UNSW Medicine South Western Sydney Clinical School at Liverpool Hospital, and a health data scientist at the Ingham Institute for Applied Medical Research, also located at Liverpool, Sydney. His background is in general medicine, general practice medicine, occupational health, public health practice, particularly population health surveillance, and clinical epidemiology.

[COVID-19 epidemiology with R by Tim Churches](https://rviews.rstudio.com/2020/03/05/covid-19-epidemiology-with-r/)

[Analysing COVID-19 (2019-nCoV) outbreak data with R - part 1](https://timchurches.github.io/blog/posts/2020-02-18-analysing-covid-19-2019-ncov-outbreak-data-with-r-part-1/#estimating-changes-in-the-effective-reproduction-number)

## IV. Results


### Model Evaluation and Validation
Given my limited understanding and knowledge in the field of epidemiology, this was the simplest model I was able to work with.

### Justification
Research is ongoing and hence there is no current benchmark for the COVID-19 epidemic. The closest data comparison is with the Spanish Flu of 1918 that infected 500 million people worldwide and killed more then 50 million people.[^9] We could possibly take its epi curve and try to fit it to the COVID-19 epidemic. Again, given my limited understanding of how epidemics work, it is challenging to calculate the epi curve for the Spanish Flu. For this reason, it is not possible to predict how how any model for COVID-19 epidemic would turn out.


## V. Conclusion


### Reflection
Based on my observations, the dataframes for **North Dakota** and **Delaware** fit the epidemic curve well. 
![enter image description here](/Images/Capture5.JPG)
![enter image description here](/Images/Capture6.JPG)
With respect to North Dakota, they have 3, 313 total cases, 2, 952 recovered cases with only 77 deaths. If they continue on this path, they are predicted to reach equilibrium (zero new cases) by the end of July or August 2020.

Delaware is another State that is following the epi curve closely. Though they had some peaks around Memorial Day week, they seem to have recovered well. If they continue on this trend, they are predicted to reach minimum amount of cases by the end of July or August 2020.

The only challenge with these predictions is that it does not take into account the human factors. Human factors can be defined as human interactions in relation to their environment, such as not following social distance measures; not wearing masks when going out in public areas; not following proper safety and sanitization rules; non-essential travel from one city to another to visit family, friends or to go on a vacation; and more. This will increase the rate of transmission leading to peaks in new cases as was witnessed around the Memorial Day week.

### Improvement
1. Using  `earlyR`  and  `EpiEstim`  packages which is part of R.
2. Partnering up with an epidemiologist to a better understanding of the COVID-19 epidemic. 

### Final Thoughts
This project has taught me a lot about machine learning and how epidemiologists are using machine learning. There were several lessons to be learnt but the top three that stood out for me are:
1.	Work with the right expert: My understanding of epidemiology is limited. I would have appreciated an opportunity to work with an epidemiologist to understand how an epidemic works in order to apply the nuances of the data to machine learning. It is really critical to understand the story behind the data to be able to build a good machine model.

2.	Generalize the problem: My proposal was confined to a narrow dataframe and solution that did not leave any room for improvisation. This led to a tunnel vision when trying to build the data model. An alternative would have been generalizing the problem in the proposal which would have allowed me to manoeuvre in different ways to come up with innovative solutions.

3.	Time management: There is a ton of published papers and research available that provide top-notch information. Going forward, I will allow myself sufficient time to go through available research before embarking on the solution. Some good sources of information are Google Scholar, Medium, GitHub repositories, and other open-source packages & libraries.
Hindsight is indeed 2020. Armed with this knowledge, I am confident that I can continue to apply myself in the field of machine learning to find novel solutions to human challenges.


To conclude, I would like to thank the following people without whom I would not have been able to complete my project and get an understanding of how COVID-19 is being used in machine learning:

 -  [Dr. Tim Churches](https://theconversation.com/profiles/timothy-churches-1003068). 
 - [Dr. Jason Brownlee](https://machinelearningmastery.com/about/)
 -  [Mauro Di Pietro](https://medium.com/@m.dipietro09)
 - [Subhasree Chatterjee](https://datascienceplus.com/author/chatterjee-subhasree/)

And last but not least, my wife [Shamsia Quraishi](https://twitter.com/shamsiaquraishi) for supporting me during my training and being there for me. 

Once again thanks and be safe.   

-----------
**Endnotes**

[^1]:[COVID-19 Dashboard](https://systems.jhu.edu/research/public-health/ncov/) by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU)". ArcGIS. Johns Hopkins University. Retrieved 20 June 2020.

[^2]:[Logistic Growth Model for COVID-19](https://www.wolframcloud.com/obj/covid-19/Published/Logistic-Growth-Model-for-COVID-19.nb)

[^3]:[Mathematical prediction of the time evolution of the COVID-19 pandemic in Italy by a Gauss error function and Monte Carlo simulations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7156796/)

[^4]:[Time Series Forecasting with Parametric Curve Fitting](https://medium.com/analytics-vidhya/how-to-predict-when-the-covid-19-pandemic-will-stop-in-your-country-with-python-d6fbb2425a9f)

[^5]:[Best Practices for Using the DeepAR Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html#deepar_best_practices)

[^6]:[11 Classical Time Series Forecasting Methods in Python (Cheat Sheet)](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)

[^7]:[Autoregressive Conditional Heteroskedasticity (ARCH)](https://www.investopedia.com/terms/a/autoregressive-conditional-heteroskedasticity.asp)

[^8]:[Autoregressive Integrated Moving Average Model with exogenous variables (ARIMAX) transfer function model for Sharpes Ferry Well and Silver Springs](https://rstudio-pubs-static.s3.amazonaws.com/180268_8acc87dd9fa2435c8a8e5ed6b815be2c.html)

[^9]:[Generalized autoregressive conditional heteroskedasticity](https://www.sciencedirect.com/science/article/abs/pii/0304407686900631?via%3Dihub)


[^7]:[# Logistic growth modelling of COVID-19 proliferation in China and its international implications](https://www.sciencedirect.com/science/article/pii/S1201971220303039)

[Covid-19 predictions using a Gauss model, based on data from April 2](https://www.preprints.org/manuscript/202004.0175/v1/download)

[^8]:[Time Series Forecasting with Parametric Curve Fitting](https://medium.com/analytics-vidhya/how-to-predict-when-the-covid-19-pandemic-will-stop-in-your-country-with-python-d6fbb2425a9f)

[^9]:[Compare: 1918 Spanish Influenza Pandemic Versus COVID-19](https://www.biospace.com/article/compare-1918-spanish-influenza-pandemic-versus-covid-19/)


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4MjE2NDA5OTMsMTcxNzUwMTU0OCwtMT
QzNTg4MzA2MiwtNjY0Njg0OTc4LC05ODEzNjA4MzYsLTEzNzYy
ODY3NTMsNDcyNDM2NDE0LC0xNTY2NDI5MTcxLDE3MjAzMzQzMy
wtNzc1NDIxMzg3LC01Nzk2NjA5MzcsNTY4MjcwNDMxLC0xNjg1
OTIwMDY1LDE0OTM4ODg4MzEsLTE4OTc5MDg5NjMsMTczOTQzMj
UxMiwxOTg3OTk3Njk2LDM1MTIxNTgxLC00OTExODg0OCw1ODIz
OTM3NjVdfQ==
-->