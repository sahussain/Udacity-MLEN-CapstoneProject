# Machine Learning Engineer Nanodegree
## Capstone Proposal

Ashraf Hussain
June 19th, 2020


## Proposal

Forecasting COVID-19 Cases – A DeepAR Model

### Domain Background

On December 31, 2019, the World Health Organization (WHO) was informed of an outbreak of “pneumonia of unknown cause” detected in Wuhan City, Hubei Province, China. Identified as coronavirus disease 2019, it quickly came to be known as COVID-19 and has resulted in an ongoing global pandemic. As of 20 June 2020, more than 8.74 million cases have been reported across 188 countries and territories, resulting in more than 462,000 deaths. More than 4.31 million people have recovered .
In response to this ongoing public health emergency, Johns Hopkins University (JHU), a private research university in Maryland, USA, developed an interactive web-based dashboard hosted by their Center for Systems Science and Engineering (CSSE). The dashboard visualizes and tracks reported cases in real-time, illustrating the location and number of confirmed COVID-19 cases, deaths and recoveries for all affected countries. It is used by researchers, public health authorities, news agencies and the general public. All the data collected and displayed is made freely available in a GitHub repository.


### Problem Statement

This project seeks to forecast number of people infected and number of deaths caused by COVID-19 for a time duration of 14-days based on historical data from JHU. I will be using Amazon SageMaker DeepAR forecasting algorithm, a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNN) .
DeepAR is an underutilized approach in this area . The dataset contains hundreds of related time series, and DeepAR outperforms classical forecasting methods autoregressive integrated moving average (ARIMA) or exponential smoothing (ETS), for this type of applications.


### Datasets and Inputs

The datasets are accessed from two source files provided by the JHU GitHub repository:
- [time_series_covid19_confirmed_US.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv "time_series_covid19_confirmed_US.csv")
- [time_series_covid19_deaths_US.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv "time_series_covid19_deaths_US.csv")

Both files have the same columns:
* UID - UID = 840 (country code3) + 000XX (state FIPS code). Ranging from 8400001 to 84000056.
* iso2-  Officially assigned country code identifiers 2 Chr (US, CA, ...)
* iso3 - Officially assigned country code identifiers 3 Chr.(USA, CAN, ...)
* code3- country code USA = 840
* FIPS -Federal Information Processing Standards code that uniquely identifies counties within the USA.
* admin2 - County name. US only.
* Province_State - The name of the State within the USA.
* Country_Region - The name of the Country (US).
* Combined_Key -  Province_State + Country_Region 
* Population - Population
* Number of cases/deaths are is columns where each column is a day


### Solution Statement
_(approx. 1 paragraph)_

 The solution will be predictions of next 14 day cases and deaths. as it is the current Incubation period.[^1] I will have to look at what context length to set at. At this point I will set that at 28 Days.
[<img src="https://github.com/sahussain/ML_SageMaker_Studies/blob/master/Time_Series_Forecasting/notebook_ims/context_prediction_windows.png">](https://github.com/sahussain/ML_SageMaker_Studies/blob/master/Time_Series_Forecasting/notebook_ims/context_prediction_windows.png)

Source: [ML_SageMaker_Studies by udacity]([https://newsinteractives.cbc.ca/coronaviruscurve/](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Time_Series_Forecasting/notebook_ims))
 
Since the data sets are relevantly clean I expect to spend 50% of the time on data cleaning and DeepAR processing part and 50% of the time on training models and tweaking parameters.


### Benchmark Model
_(approximately 1-2 paragraphs)_

For this problem, the benchmark model will for both the number of cases and deaths will be to be in between ±80% confidence interval 

### Evaluation Metrics
_(approx. 1-2 paragraphs)_


?????


### Project Design
_(approx. 1 page)_


First we need to convert the data form wide table format to tall table format. The next thing to do will be to see what `cardinality` data to include. Once that is done I will be use DeepAR Forecasting Algorithm to forecast the next 14-day view for both cases and deaths.


-----------


**Before submitting your proposal, ask yourself. . .**
- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?

# Endnotes
[^1]: [WHO. Coronavirus disease 2019 (COVID-19) Situation Report -59. [Online] 20 March 2020](https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200319-sitrep-59-covid-19.pdf?sfvrsn=c3dcdef9_2)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMTE3MDk2MDAsMTY1MzIyODAzNCwtMT
QwNTg1NDIyNiwzNjM2OTA1NjYsMTQ0NzY2NzQ0NiwxMzgzMjky
MjQyLDE2MzE2MTIzODAsLTE2ODA3MjQxMiwtODkwNDU2OTAsLT
gwMzM1MTE5MCwtOTgxMTUwMzAsLTIwMDQ5NDg1OTEsMTYwODc2
ODU2OCwxMjY5MDU1NDgwLDEyMTU4MDU4ODgsLTE5NjIyNDc1MT
csLTE3MTcxMDUzNTZdfQ==
-->