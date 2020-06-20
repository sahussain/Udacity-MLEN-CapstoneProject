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

The goal of this project is to predict if we are **"flattening the curve"**. Each state is slowly lifting there **stay at home order** and **easing social distancing measures**. I would be looking at forecasting the next 14 day as it is the current Incubation period.[^1]  

Little bit info on what is **"flattening the curve"**?:
>An epidemic curve, or “epicurve,” is a graph that shows the frequency of new cases over time based on new infections per day. In many cases, an epicurve follows a bell curve, steadily rising to a peak and then declining as the outbreak burns out when the virus runs out of people to infect.
>
>Flattening the curve refers to when methods like large-scale testing, quarantining of infected individuals and social distancing are used to decrease the number of daily new COVID-19 cases. The aim is to reduce overall infections and keep cases at a number the health-care system can manage.

## Flattening the curve
![](Images/Flatteningthecurve.svg)



Source: [CBC]([https://newsinteractives.cbc.ca/coronaviruscurve/](https://newsinteractives.cbc.ca/coronaviruscurve/))


### Datasets and Inputs
_(approx. 2-3 paragraphs)_


In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.


### Solution Statement
_(approx. 1 paragraph)_


In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).


### Benchmark Model
_(approximately 1-2 paragraphs)_


In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.


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
eyJoaXN0b3J5IjpbMTkyNDQwNDYxNywtODAzMzUxMTkwLC05OD
ExNTAzMCwtMjAwNDk0ODU5MSwxNjA4NzY4NTY4LDEyNjkwNTU0
ODAsMTIxNTgwNTg4OCwtMTk2MjI0NzUxNywtMTcxNzEwNTM1Nl
19
-->