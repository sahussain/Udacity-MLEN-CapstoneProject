# Machine Learning Engineer Nanodegree
## Capstone Proposal
Author
: Ashraf Hussain

Date 
: June 19th, 2020


## Proposal
[COVID-19 Data Research repository from Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

### Domain Background
The data is publicly available form  Johns Hopkins University (JHU) which is a teaching and research hospital. They believe: 
> "that sharing our **knowledge** and **discoveries** would help make the world a better place"  

The first president of JHU Daniel Coit Gilman summarizes what JHU is all about, when he was asked 
>"[his inauguration in 1876](https://www.jhu.edu/about/history/gilman-address/). What is this place all about, exactly? His answer: 
>_“The encouragement of research . . . and the advancement of individual scholars, who by their excellence will advance the sciences they pursue, and the society where they dwell.”_

Since the start for the [COVID-19]([https://en.wikipedia.org/wiki/Coronavirus_disease_2019](https://en.wikipedia.org/wiki/Coronavirus_disease_2019)) pandemic JHU have been one of the critical source in providing data both to the public and researches alike. There data is being used by many news outlets as well like [CNN]([https://www.cnn.com/interactive/2020/health/coronavirus-us-maps-and-cases/](https://www.cnn.com/interactive/2020/health/coronavirus-us-maps-and-cases/)), [CBC]([https://newsinteractives.cbc.ca/coronavirustracker/](https://newsinteractives.cbc.ca/coronavirustracker/)), etc.

They have a [JHU Git repository](git@github.com:CSSEGISandData/COVID-19.git) which provides the data in many different views, I will be using there [csse_covid_19_time_series](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series "csse_covid_19_time_series") 

### Problem Statement
_(approx. 1 paragraph)_

The goal of this project two fold 
1. Is to determine the 14-day projected data is showing sings of **"flattening the curve"**.
2. What is the 14-day projected data is showing for the number of deaths. 

Little bit info on what is **"flattening the curve"**?:
>An epidemic curve, or “epicurve,” is a graph that shows the frequency of new cases over time based on new infections per day. In many cases, an epicurve follows a bell curve, steadily rising to a peak and then declining as the outbreak burns out when the virus runs out of people to infect.
>
>Flattening the curve refers to when methods like large-scale testing, quarantining of infected individuals and social distancing are used to decrease the number of daily new COVID-19 cases. The aim is to reduce overall infections and keep cases at a number the health-care system can manage.

The criteria that will be use to determine if there is a flattening the curve based on if the following holds:
1. if the predicted line of confirmed cases is trending towers straight line
2. if the predicted line of number of deaths is trending lower


## Flattening the curve
[<img src="https://github.com/sahussain/Udacity-MLEN-CapstoneProject/blob/master/Images/Flatteningthecurve.svg">](https://github.com/sahussain/Udacity-MLEN-CapstoneProject/blob/master/Images/Flatteningthecurve.svg)

Source: [CBC]([https://newsinteractives.cbc.ca/coronaviruscurve/](https://newsinteractives.cbc.ca/coronaviruscurve/))


### Datasets and Inputs
_(approx. 2-3 paragraphs)_
There will be two data source files:
- [time_series_covid19_confirmed_US.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv "time_series_covid19_confirmed_US.csv")
- [time_series_covid19_deaths_US.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv "time_series_covid19_deaths_US.csv")


### Solution Statement
_(approx. 1 paragraph)_

 The solution will be predictions of next 14 day cases and deaths. as it is the current Incubation period.[^1] I will have to look at what context length to set at. At this point I will set that at 28 Days.
[<img src="https://github.com/sahussain/ML_SageMaker_Studies/blob/master/Time_Series_Forecasting/notebook_ims/context_prediction_windows.png">](https://github.com/sahussain/ML_SageMaker_Studies/blob/master/Time_Series_Forecasting/notebook_ims/context_prediction_windows.png)

Source: [ML_SageMaker_Studies by udacity]([https://newsinteractives.cbc.ca/coronaviruscurve/](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Time_Series_Forecasting/notebook_ims))
 
Since the data sets are relevantly clean I expect to spend 50% of the time on data cleaning and DeepAR processing part and 50% of the time on training models and tweaking parameters.


### Benchmark Model
_(approximately 1-2 paragraphs)_

For this problem, the benchmark model will for both the number of cases and deaths will be to be in between ±80% 

### Evaluation Metrics
_(approx. 1-2 paragraphs)_


In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).


### Project Design
_(approx. 1 page)_


In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.


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
eyJoaXN0b3J5IjpbLTE3MDE3MDY2MTksMzYzNjkwNTY2LDE0ND
c2Njc0NDYsMTM4MzI5MjI0MiwxNjMxNjEyMzgwLC0xNjgwNzI0
MTIsLTg5MDQ1NjkwLC04MDMzNTExOTAsLTk4MTE1MDMwLC0yMD
A0OTQ4NTkxLDE2MDg3Njg1NjgsMTI2OTA1NTQ4MCwxMjE1ODA1
ODg4LC0xOTYyMjQ3NTE3LC0xNzE3MTA1MzU2XX0=
-->