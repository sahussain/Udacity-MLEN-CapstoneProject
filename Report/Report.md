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

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

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
[^8]:
[^9]:
[^10]:
[^11]:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExODAzMDUyMTgsMTM0MjcyMjM0MywtNz
Y3NTYxMzE2LDU1ODc5ODA3NCwtMTUwNzUyMjU0MCwtNDg1NzE1
NDk0LDEyOTk5MjMyOSwtNDA4MDE2OTY3LC0xMTc5NDk1MDkwLD
Y1MjExMzk0NSwtMTU4MTIxMTEzNywtMTkyNjQ0ODM4LC03NzA5
MDQ4MzVdfQ==
-->