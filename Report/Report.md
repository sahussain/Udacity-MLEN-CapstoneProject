# Machine Learning Engineer Nanodegree
## Capstone Project
Ashraf Hussain 
June 22th, 2020

Machine Learning Engineer Nanodegree: Forecasting COVID-19 Cases – A Time Series Forecasting Model

## I. Definition


### Project Overview

On December 31, 2019, the World Health Organization (WHO) was informed of an outbreak of “pneumonia of unknown cause” detected in Wuhan City, Hubei Province, China. Identified as coronavirus disease 2019, it quickly came
to be known as COVID-19 and has resulted in an ongoing global pandemic. As
of 20 June 2020, more than 8.74 million cases have been reported across
188 countries and territories, resulting in more than 462,000 deaths. More
than 4.31 million people have recovered.[^1]

In response to this ongoing public health emergency, Johns Hopkins
University (JHU), a private research university in Maryland, USA,
developed an interactive web-based dashboard hosted by their Center for
Systems Science and Engineering (CSSE). The dashboard visualizes and
tracks reported cases in real-time, illustrating the location and number
of confirmed COVID-19 cases, deaths and recoveries for all affected
countries. It is used by researchers, public health authorities, news
agencies and the general public. All the data collected and displayed is
made freely available in a [GitHub repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data).

I was going to use going to use [DeepAR by AWS]([https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)) and compare it to  AR (autoregressive model). After exploring the data the data I discovered that this data set can not be used for any Time Series Forecasting Model(TSFMs) or any other ML models. 

This is because each epidemic of total number of cases follows Logistic Function which is defined by:
 
f(x) = capacity / (1 + e^-k*(x - midpoint) )[^2]

And the epidemic of new of cases follows Gaussian Function which is defined by:
f(x) = a * e^(-0.5 * ((x-μ)/σ)**2)[^3]

### Problem Statement
This project seeks to forecast number of people infected and number of
caused by COVID-19 for a time duration of 14-days based on
historical data from JHU. I will be using Amazon SageMaker DeepAR
forecasting algorithm, a supervised learning algorithm for forecasting
scalar (one-dimensional) time series using recurrent neural networks (RNN)
to produce both point and probabilistic forecasts[^4].
DeepAR is an underutilized approach in this area.[^5] The dataset contains
hundreds of related time series, and DeepAR outperforms classical
forecasting methods including but not limited to autoregressive integrated
moving average (ARIMA), exponential smoothing (ETS), Time Series
Forecasting with Linear Learner for this type of applications.

Since epidemic do not follow a standard time series requirements however they do follow Logistic & Gaussian functions what we have to do is to fit our total cases to Logistic function and new cases to Gaussian functions. 


### Metrics
The error represents random variations in the data that follow a specific probability distribution (usually Gaussian). The objective of curve fitting is to find the optimal combination of parameters that minimize the error. Here we are dealing with time series, therefore the independent variable is time. In mathematical terms[^6]
f(error) = f(time) + error


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
The plot below shows how the COVID-19 cases incise by city
When looking at the graph of 15 cities I noticed the first problem. The dataset only have data for the last 151 days.

I can not use DeepAR on this data because DeepAR needs at least 300 observations available across all training time series and we have at most 151.

>We recommend training a DeepAR model on as many time series as are available. Although a DeepAR model trained on a single time series might work well, standard forecasting algorithms, such as ARIMA or ETS, might provide more accurate results. The DeepAR algorithm starts to outperform the standard methods when your dataset contains hundreds of related time series. Currently, DeepAR requires that the total number of observations available across all training time series is at least 300. Source: https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html

![enter image description here](/Images/Capture2.JPG)

Now lets look and new vs total for some cities:
![enter image description here](/Images/Capture3.JPG)
![enter image description here](/Images/Capture4.JPG)





### Algorithms and Techniques
I will be using [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) function, part of [SciPy](https://scipy.org/)
Peak fitting with a Gaussian/Logistic Function is very commonly used in epidemiology. [^7]

### Benchmark
Since we are still in the early stage of epidemiology, there is no real bench mark, however the model was able to predict future cases for some state very well. however other states it could not I looked at some random cities.
**Delaware**
![enter image description here](/Images/Capture5.JPG)

Delaware seem to be doing very well and there social-distancing measures seem to be working, They did had some peaks of new cased during Memorial Day. They did seem to recovered from it very quickly.   

**North Dakota**
![enter image description here](/Images/Capture6.JPG)
North Dakota seem to be doing very well too and there social-distancing measures seem to be working, They did had some peaks of new cased during Memorial Day same as  Delaware however they took little bit longer to come back to normal.

**Maryland**
![enter image description here](/Images/Capture7.JPG)
North Dakota seem to be doing very well too and there social-distancing measures seem to be working, They did had some peaks of new cased during Memorial Day however there peak was delayed compared to Delaware and North Dakota this could be because people may have wend to another state for Memorial Day. 

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
The [time_series_covid19_confirmed_US.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv) 
|**UID**|**iso2**|**iso3**|**code3**|**FIPS**|**Admin2**|**Province\_State**|**Country\_Region**|**Lat**|**Long\_**|**...**|**6/13/20**|**6/14/20**|**6/15/20**|**6/16/20**|**6/17/20**|**6/18/20**|**6/19/20**|**6/20/20**|**6/21/20**|**6/22/20**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
0|16|AS|ASM|16|60|NaN|American Samoa|US|-14.271|-170.132|...|0|0|0|0|0|0|0|0|0|0
1|316|GU|GUM|316|66|NaN|Guam|US|13.4443|144.7937|...|183|183|185|186|188|192|200|222|222|222
2|580|MP|MNP|580|69|NaN|Northern Mariana Islands|US|15.0979|145.6739|...|30|30|30|30|30|30|30|30|30|30
3|630|PR|PRI|630|72|NaN|Puerto Rico|US|18.2208|-66.5901|...|5690|5811|5890|5951|6003|6111|6195|6463|6525|6564
4|850|VI|VIR|850|78|NaN|Virgin Islands|US|18.3358|-64.8963|...|72|72|72|72|73|73|73|73|76|76
The data set was imported into a pandas Dataframe. 

Data needed minimal data Preprocessing as the each date was in a column and City, State was in another columns.
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


Then the data frame was converted to convert index to datetime
```python
## convert index to datetime
covid_df.index = pd.to_datetime(covid_df.index, infer_datetime_format=True)
```
Form hear onward we can use the following function to get a state data by name
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
The [scipy.optimize.curve_fit model  from scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) which provides with non-linear least squares to fit a function, f, to data. functions like Logistic, or Gaussian functions where the data is COVID-19 total and new cases. 
Forecast:
The forecasting function was provided by [`ts_utils.py`](https://github.com/mdipietro09/DataScience_ArtificialIntelligence_Utils/blob/master/time_series/ts_utils.py)
which apply the 2 models(total cases data and one for the daily increase) to a new independent variable: the time steps from today till N. It forecast 30 days ahead from today[^8]


### Refinement
The model and the forecast works well however there are few states that will not work because they are on a different epidemic curve for example **Michigan**
![enter image description here](/Images/Capture9.JPG)
The model could not fit Total Cases to a Logistic Function

**Virgin Islands**
![enter image description here](/Images/Capture11.JPG)
The model could not fit new Cases to a Gaussian Function

A better tool kit like  `earlyR`  and  `EpiEstim`  packages which is part of R. 
There are two good articles talk about how to use R to predict COVID cases both of them was written by [Tim Churches](https://theconversation.com/profiles/timothy-churches-1003068). 
>Tim Churches is a Senior Research Fellow at the UNSW Medicine South Western Sydney Clinical School at Liverpool Hospital, and a health data scientist at the Ingham Institute for Applied Medical Research, also located at Liverpool, Sydney. His background is in general medicine, general practice medicine, occupational health, public health practice, particularly population health surveillance, and clinical epidemiology.

[COVID-19 epidemiology with R by Tim Churches](https://rviews.rstudio.com/2020/03/05/covid-19-epidemiology-with-r/):
[Analysing COVID-19 (2019-nCoV) outbreak data with R - part 1]([https://timchurches.github.io/blog/posts/2020-02-18-analysing-covid-19-2019-ncov-outbreak-data-with-r-part-1/#estimating-changes-in-the-effective-reproduction-number](https://timchurches.github.io/blog/posts/2020-02-18-analysing-covid-19-2019-ncov-outbreak-data-with-r-part-1/#estimating-changes-in-the-effective-reproduction-number))

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
This was the simplest model I could pick without me getting a degree in epidemiology

### Justification
There is no current benchmark to COVID-19 so we don't know how any model would turned out. The only one we can compare is to Spanish Flu of 1918 Compared which infected 500 million people worldwide and killed more then 50 million people.[^9] We could possibly take its  epidemic curve and try to fit it to COVID-19. Again its beyond my understanding to even calculate the epidemic curve for the Spanish Flu. 


## V. Conclusion
_(approx. 1-2 pages)_

### Reflection
I would say my best model would be **North Dakota** and **Delaware**
both of the states fit the epidemic curve. 
![enter image description here](/Images/Capture5.JPG)
![enter image description here](/Images/Capture6.JPG)
With respect to North Dakota They have 3313 cases 2952 recovered with only 77 debts they are doing very well.  If they continue  on this path they will have reached zero new cases around the end of July or August. 
Delaware is another state that is following the epidemic curve pretty closely. even though they had some pigs around the Memorial Day week and they seem to have recovered pretty well if they continue to on this trend they will reach minimum amount of cases by end of July or August

The only problem with these predictions cannot take into account the human factor. The  human factor that is referring to is where people travel from one city to another to visit family, friends or to go on a vacation.  This could increase the transmission rate and you will see peaks, In the number of new cases during these periods case and point Memorial Day weekend peaks. 

### Improvement
1. Using  `earlyR`  and  `EpiEstim`  packages which is part of R.
2. Partnering up with an epidemiologist to get the understanding of the problem. 

### Final Thoughts
This project has taught me a lot about machine learning and epidemiology and how epidemiologist are using machine learning. This project has also taught me that we always need a partner in this case it would have been really nice if your I would have had an epidemiologist to help me and guide me in the process of how a disease works. We as machine learning Engineers only understands the data but for us to be able to program a machine to learn something new we need to also understand what is the data based on what are the nuances on the data itself what is the story behind the data without the this. 

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
-----------
**Endnotes**
-----------

[^1]:[COVID-19 Dashboard](https://systems.jhu.edu/research/public-health/ncov/) by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU)". ArcGIS. Johns Hopkins University. Retrieved 20 June 2020.
[^2]:[Logistic Growth Model for COVID-19](https://www.wolframcloud.com/obj/covid-19/Published/Logistic-Growth-Model-for-COVID-19.nb)
[^3]:[Mathematical prediction of the time evolution of the COVID-19 pandemic in Italy by a Gauss error function and Monte Carlo simulations]([https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7156796/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7156796/))
[^4]:[DeepAR Forecasting Algorithm. Amazon Web Services](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html). Retrieved 20 June, 2020 
[^5]:[Time series prediction](https://www.telesens.co/2019/06/08/time-series-prediction/). Telesens. Retrieved 20 June, 2020.
[^6]:[Time Series Forecasting with Parametric Curve Fitting](https://medium.com/analytics-vidhya/how-to-predict-when-the-covid-19-pandemic-will-stop-in-your-country-with-python-d6fbb2425a9f)
[^7]:[# Logistic growth modelling of COVID-19 proliferation in China and its international implications](https://www.sciencedirect.com/science/article/pii/S1201971220303039)
[Covid-19 predictions using a Gauss model, based on data from April 2](https://www.preprints.org/manuscript/202004.0175/v1/download)
[^8]:[Time Series Forecasting with Parametric Curve Fitting](https://medium.com/analytics-vidhya/how-to-predict-when-the-covid-19-pandemic-will-stop-in-your-country-with-python-d6fbb2425a9f)
[^9]:[Compare: 1918 Spanish Influenza Pandemic Versus COVID-19](https://www.biospace.com/article/compare-1918-spanish-influenza-pandemic-versus-covid-19/)
[^10]:
[^11]:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY0NDQyNDQzMSwxOTg3OTk3Njk2LDM1MT
IxNTgxLC00OTExODg0OCw1ODIzOTM3NjUsMTg5NjcxNTk1OSwx
MzQyNzIyMzQzLC03Njc1NjEzMTYsNTU4Nzk4MDc0LC0xNTA3NT
IyNTQwLC00ODU3MTU0OTQsMTI5OTkyMzI5LC00MDgwMTY5Njcs
LTExNzk0OTUwOTAsNjUyMTEzOTQ1LC0xNTgxMjExMTM3LC0xOT
I2NDQ4MzgsLTc3MDkwNDgzNV19
-->