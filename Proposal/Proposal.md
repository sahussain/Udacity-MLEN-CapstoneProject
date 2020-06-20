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
<svg width="750" height="594" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 594" aria-label="Graphic showing a steep curve in number of cases without intervention and a flatter curve with intervention">
  <defs>
    <style>
      .cls-1,.cls-7{isolation:isolate}.cls-7{font-size:16px;font-family:OpenSans-Regular,Open Sans}.cls-8{fill:#677279}.cls-7{fill:#677279}.cls-9{fill:#fff;stroke:none;}.cls-10{fill:#667279;stroke:none;}
    </style>
  </defs>
  <g class="cls-1">
    <g id="Layer_1" data-name="Layer 1">
      <g id="capacity">
        <path id="over" d="M460 0c118.88 0 93 560 439 560H21C362.67 560 341.12 0 460 0z" fill-rule="evenodd" fill="#e0484d"></path>
      </g>
      <g id="capacity-2" data-name="capacity">
        <path id="under" d="M650.08 339.77C863.66 339.77 865.06 560 1280 560H21c432.28 0 415.5-220.23 629.08-220.23z" style="mix-blend-mode:multiply" fill="#577d82" fill-rule="evenodd"></path>
        <path id="stroke" d="M650.5 339.77C864.08 339.77 865.06 560 1280 560H21c432.28 0 415.92-220.23 629.5-220.23z" stroke-width="2" stroke-miterlimit="10" stroke="#fff" fill="none"></path>
      </g>
      <g id="threshold">
        <text id="label" transform="translate(864 313.03)" class="cls-7" style="isolation:isolate" fill="#677279" font-size="16" font-family="OpenSans-Regular,Open Sans">
          Healthcare system capacity
        </text>
        <path id="threshold-line" style="mix-blend-mode:luminosity" stroke="#979797" stroke-dasharray="3 2" fill="none" d="M37 326.13h1243"></path>
      </g>
      <text id="label-2" data-name="label" class="cls-7" transform="matrix(0 -1 1 0 13.71 560)">
        Number of cases
      </text>
      <path id="arrow-up" class="cls-8" d="M1208.4 585l-18 9v-8H211v-2h979.4v-8z" fill="#677279"></path>
      <path id="arrow-up-2" data-name="arrow-up" class="cls-8" d="M10 0l9 18h-8v392H9V18H1z" fill="#677279"></path>
      <text id="label-3" data-name="label" class="cls-7" transform="translate(37 589.69)">
        Days since first case
      </text>
      <g class="cls-1">
        <path class="cls-9" d="M723.07 450.08a3.32 3.32 0 01-1.17 2.74 5.15 5.15 0 01-3.34 1h-1.18v4.27h-1.87v-11.46h3.29a5 5 0 013.2.87 3.14 3.14 0 011.07 2.58zm-5.69 2.13h1a3.58 3.58 0 002.11-.5 1.86 1.86 0 00.67-1.57 1.8 1.8 0 00-.6-1.47 3 3 0 00-1.88-.48h-1.29zM730.56 458.05l-.36-1.2h-.07a3.53 3.53 0 01-1.25 1.07 4 4 0 01-1.63.29 2.75 2.75 0 01-2-.69 2.61 2.61 0 01-.71-1.95 2.27 2.27 0 011-2 5.72 5.72 0 013-.75h1.49v-.46a1.72 1.72 0 00-.38-1.24 1.56 1.56 0 00-1.2-.41 4.11 4.11 0 00-1.27.19 11.11 11.11 0 00-1.18.46l-.59-1.31a6.7 6.7 0 011.54-.56 7.55 7.55 0 011.58-.19 3.74 3.74 0 012.47.7 2.85 2.85 0 01.84 2.26v5.82zm-2.73-1.25a2.26 2.26 0 001.6-.56 2 2 0 00.61-1.57v-.75l-1.11.05a3.92 3.92 0 00-1.89.43 1.34 1.34 0 00-.59 1.19 1.15 1.15 0 00.35.89 1.48 1.48 0 001.03.32zM742 458.05h-1.8v-5.31a2.33 2.33 0 00-.41-1.5 1.57 1.57 0 00-1.27-.49 2 2 0 00-1.71.69 3.79 3.79 0 00-.54 2.3v4.31h-1.83v-8.64h1.44l.25 1.13h.1a2.53 2.53 0 011.11-1 3.63 3.63 0 011.59-.34q3.11 0 3.11 3.17zM747.51 458.21A3 3 0 01745 457a5.28 5.28 0 01-.9-3.29 5.2 5.2 0 01.92-3.3 3 3 0 012.54-1.19 3 3 0 012.59 1.26h.1a11 11 0 01-.14-1.47v-3.15h1.85v12.16h-1.46l-.32-1.13h-.09a2.93 2.93 0 01-2.58 1.32zm.49-1.49a2 2 0 001.65-.64 3.35 3.35 0 00.53-2.06v-.26a3.9 3.9 0 00-.53-2.32 2 2 0 00-1.67-.69 1.68 1.68 0 00-1.49.79 4 4 0 00-.52 2.24 3.94 3.94 0 00.51 2.19 1.71 1.71 0 001.52.75zM758.34 458.21a4.21 4.21 0 01-3.16-1.18 4.48 4.48 0 01-1.13-3.24 4.92 4.92 0 011.05-3.33 3.66 3.66 0 012.9-1.21 3.56 3.56 0 012.7 1 4 4 0 011 2.86v1h-5.76a2.81 2.81 0 00.68 1.94 2.35 2.35 0 001.8.67 7.15 7.15 0 001.43-.14 8.2 8.2 0 001.42-.48v1.49a6 6 0 01-1.36.45 8.25 8.25 0 01-1.57.17zm-.34-7.57a1.84 1.84 0 00-1.4.56 2.67 2.67 0 00-.63 1.61h3.92a2.41 2.41 0 00-.51-1.62 1.81 1.81 0 00-1.38-.55zM771.06 458.05h-1.84v-5.33a2.42 2.42 0 00-.38-1.48 1.4 1.4 0 00-1.17-.49 1.79 1.79 0 00-1.56.69 4.07 4.07 0 00-.49 2.3v4.31h-1.84v-8.64h1.44l.26 1.13h.09a2.4 2.4 0 011-1 3.32 3.32 0 011.51-.34 2.63 2.63 0 012.64 1.36h.12a2.67 2.67 0 011.08-1 3.42 3.42 0 011.59-.36 2.9 2.9 0 012.26.78 3.48 3.48 0 01.71 2.39V458h-1.84v-5.33a2.42 2.42 0 00-.38-1.48 1.41 1.41 0 00-1.18-.49 1.8 1.8 0 00-1.56.67 3.46 3.46 0 00-.5 2.05zM779 447.12a.93.93 0 011-1 1 1 0 01.75.26 1 1 0 01.27.76.93.93 0 01-1 1 1 1 0 01-.77-.27 1 1 0 01-.25-.75zm1.94 10.93h-1.83v-8.64h1.83zM787 458.21a3.81 3.81 0 01-3-1.15 4.77 4.77 0 01-1-3.28 4.84 4.84 0 011.07-3.36 4 4 0 013.08-1.17 5.76 5.76 0 012.46.51l-.55 1.48a5.45 5.45 0 00-1.92-.46c-1.5 0-2.25 1-2.25 3a3.55 3.55 0 00.56 2.18 1.93 1.93 0 001.64.73 4.65 4.65 0 002.32-.61v1.61a3.64 3.64 0 01-1 .41 6.53 6.53 0 01-1.41.11zM803.28 453.71a4.75 4.75 0 01-1.08 3.29 3.92 3.92 0 01-3 1.19 4.18 4.18 0 01-2.14-.55 3.63 3.63 0 01-1.43-1.57 5.41 5.41 0 01-.5-2.38 4.68 4.68 0 011.08-3.28 3.93 3.93 0 013-1.18 3.79 3.79 0 013 1.21 4.65 4.65 0 011.07 3.27zm-6.29 0c0 2 .74 3 2.21 3s2.19-1 2.19-3-.73-3-2.2-3a1.9 1.9 0 00-1.68.77 3.94 3.94 0 00-.51 2.23zM811.48 458.05l-.25-1.13h-.1a2.58 2.58 0 01-1.09.94 3.63 3.63 0 01-1.61.35 3.15 3.15 0 01-2.34-.79 3.21 3.21 0 01-.78-2.36v-5.65h1.85v5.33a2.29 2.29 0 00.41 1.48 1.53 1.53 0 001.27.5 2 2 0 001.7-.69 3.75 3.75 0 00.55-2.32v-4.3h1.84v8.64zM818.77 456.72a4.51 4.51 0 001.34-.21v1.38a3.6 3.6 0 01-.79.23 6.39 6.39 0 01-1 .09c-1.74 0-2.61-.92-2.61-2.75v-4.66h-1.18v-.8l1.26-.68.63-1.82h1.13v1.92H820v1.39h-2.46v4.62a1.29 1.29 0 00.33 1 1.21 1.21 0 00.9.29zM826.25 449.25a3 3 0 012.52 1.17 6.47 6.47 0 010 6.6 3.33 3.33 0 01-5.09 0h-.12l-.34 1h-1.37v-12.13h1.83v4.69h.08a2.91 2.91 0 012.49-1.33zm-.48 1.5a1.86 1.86 0 00-1.59.65 3.79 3.79 0 00-.51 2.19v.12a4.09 4.09 0 00.5 2.29 1.86 1.86 0 001.63.71 1.66 1.66 0 001.49-.78 4.14 4.14 0 00.5-2.23c0-1.97-.67-2.95-2.02-2.95zM836.19 449.25a4.51 4.51 0 01.91.08l-.18 1.71a3.59 3.59 0 00-.81-.09 2.37 2.37 0 00-1.79.72 2.61 2.61 0 00-.68 1.86v4.52h-1.84v-8.64h1.44l.24 1.52h.1a3.38 3.38 0 011.12-1.22 2.62 2.62 0 011.49-.46zM842.48 458.21a4.21 4.21 0 01-3.16-1.18 4.48 4.48 0 01-1.13-3.24 4.92 4.92 0 011-3.33 3.68 3.68 0 012.9-1.21 3.56 3.56 0 012.7 1 4 4 0 011 2.86v1h-5.76a2.81 2.81 0 00.68 1.94 2.35 2.35 0 001.8.67 7.15 7.15 0 001.43-.14 8.2 8.2 0 001.42-.48v1.49a6 6 0 01-1.36.45 8.18 8.18 0 01-1.52.17zm-.34-7.57a1.84 1.84 0 00-1.4.56 2.67 2.67 0 00-.63 1.61H844a2.41 2.41 0 00-.51-1.62 1.81 1.81 0 00-1.35-.55zM853.33 458.05l-.37-1.2h-.06a3.64 3.64 0 01-1.26 1.07 4 4 0 01-1.62.29 2.74 2.74 0 01-2-.69 2.57 2.57 0 01-.72-1.95 2.27 2.27 0 011-2 5.76 5.76 0 013-.75h1.49v-.46a1.72 1.72 0 00-.39-1.24 1.53 1.53 0 00-1.2-.41 4.16 4.16 0 00-1.27.19 10.17 10.17 0 00-1.17.46l-.56-1.36a6.61 6.61 0 011.53-.56 7.63 7.63 0 011.58-.19 3.74 3.74 0 012.49.72 2.81 2.81 0 01.84 2.26v5.82zm-2.74-1.25a2.29 2.29 0 001.61-.56 2 2 0 00.6-1.57v-.75l-1.1.05a3.86 3.86 0 00-1.89.43 1.34 1.34 0 00-.59 1.19 1.15 1.15 0 00.34.89 1.49 1.49 0 001.03.32zM859 453.49l1-1.3 2.6-2.78h2.12l-3.47 3.71 3.69 4.93h-2.14l-2.8-3.79-1 .83v3h-1.8v-12.2h1.8v5.93l-.09 1.67zM866.08 450.4a1 1 0 011.15-1.18 1.18 1.18 0 01.88.31 1.45 1.45 0 010 1.75 1.11 1.11 0 01-.87.32 1.07 1.07 0 01-.85-.32 1.22 1.22 0 01-.31-.88zm0 6.67a1.18 1.18 0 01.3-.87 1.1 1.1 0 01.85-.31 1.15 1.15 0 01.87.32 1.2 1.2 0 01.3.86 1.24 1.24 0 01-.3.88 1.12 1.12 0 01-.87.33 1.1 1.1 0 01-.85-.32 1.26 1.26 0 01-.3-.89zM722.4 477.25l-1.12-4c-.13-.43-.38-1.42-.73-3h-.07c-.31 1.41-.55 2.4-.72 3l-1.15 4h-2l-2.42-8.64H716l1.1 4.26c.25 1.05.43 1.95.53 2.7h.05c0-.38.13-.82.24-1.31s.2-.86.28-1.1l1.31-4.55h2l1.28 4.55c.08.25.17.64.29 1.17a8.41 8.41 0 01.21 1.22h.06a23.67 23.67 0 01.55-2.68l1.11-4.26h1.85l-2.44 8.64zM728.27 466.32a1 1 0 01.27-.76 1 1 0 01.77-.26.93.93 0 011 1 1 1 0 01-.27.74 1 1 0 01-.76.27 1 1 0 01-.77-.27 1 1 0 01-.24-.72zm2 10.93h-1.84v-8.64h1.84zM736.06 475.92a4.52 4.52 0 001.35-.21v1.38a3.77 3.77 0 01-.79.23 5.39 5.39 0 01-1 .09c-1.73 0-2.6-.92-2.6-2.75V470h-1.18v-.81l1.26-.68.63-1.82h1.13v1.92h2.46V470h-2.46v4.62a1.29 1.29 0 00.33 1 1.17 1.17 0 00.87.3zM746.73 477.25h-1.84v-5.31a2.34 2.34 0 00-.4-1.5 1.59 1.59 0 00-1.28-.49 2 2 0 00-1.7.69A3.84 3.84 0 00741 473v4.29h-1.84v-12.2H741v3.09a14.92 14.92 0 01-.09 1.58h.09a2.57 2.57 0 011-1 3.31 3.31 0 011.55-.35c2.1 0 3.14 1.06 3.14 3.17zM753.34 466.32a.93.93 0 011-1 1 1 0 01.75.26 1 1 0 01.27.76.93.93 0 01-1 1 1 1 0 01-.77-.27 1 1 0 01-.25-.75zm1.94 10.93h-1.83v-8.64h1.83zM765.51 477.25h-1.85v-5.31a2.34 2.34 0 00-.4-1.5A1.58 1.58 0 00762 470a2 2 0 00-1.7.69 3.79 3.79 0 00-.54 2.3v4.31h-1.83v-8.64h1.43l.26 1.13h.1a2.56 2.56 0 011.1-.95 3.67 3.67 0 011.6-.34q3.11 0 3.11 3.17zM771.28 475.92a4.41 4.41 0 001.34-.21v1.38a3.49 3.49 0 01-.78.23 5.39 5.39 0 01-1 .09c-1.74 0-2.61-.92-2.61-2.75V470h-1.18v-.81l1.27-.68.63-1.82h1.13v1.92h2.46V470h-2.46v4.62a1.29 1.29 0 00.33 1 1.17 1.17 0 00.87.3zM778.12 477.41a4.2 4.2 0 01-3.15-1.18 4.44 4.44 0 01-1.13-3.24 4.92 4.92 0 011-3.33 3.66 3.66 0 012.9-1.21 3.56 3.56 0 012.7 1 4 4 0 011 2.86v1h-5.75a2.81 2.81 0 00.68 1.94 2.34 2.34 0 001.8.67 7.15 7.15 0 001.43-.14 7.84 7.84 0 001.41-.48v1.49a5.79 5.79 0 01-1.35.45 8.29 8.29 0 01-1.54.17zm-.33-7.57a1.84 1.84 0 00-1.4.56 2.67 2.67 0 00-.63 1.61h3.92a2.37 2.37 0 00-.52-1.62 1.78 1.78 0 00-1.37-.55zM788 468.45a4.62 4.62 0 01.92.08l-.18 1.71a3.59 3.59 0 00-.81-.09 2.36 2.36 0 00-1.79.72 2.57 2.57 0 00-.68 1.86v4.52h-1.84v-8.64H785l.24 1.52h.09a3.47 3.47 0 011.12-1.22 2.64 2.64 0 011.55-.46zM792.44 477.25l-3.28-8.64h1.93l1.76 5a10.11 10.11 0 01.55 2.05h.06a13.85 13.85 0 01.55-2.05l1.76-5h2l-3.3 8.64zM802.8 477.41a4.2 4.2 0 01-3.15-1.18 4.44 4.44 0 01-1.13-3.24 4.92 4.92 0 011.05-3.33 3.66 3.66 0 012.9-1.21 3.56 3.56 0 012.7 1 4 4 0 011 2.86v1h-5.75a2.81 2.81 0 00.68 1.94 2.34 2.34 0 001.8.67 7.15 7.15 0 001.43-.14 7.84 7.84 0 001.41-.48v1.49a5.79 5.79 0 01-1.35.45 8.29 8.29 0 01-1.59.17zm-.33-7.57a1.84 1.84 0 00-1.4.56 2.55 2.55 0 00-.63 1.61h3.92a2.37 2.37 0 00-.52-1.62 1.78 1.78 0 00-1.37-.55zM815.85 477.25H814v-5.31a2.4 2.4 0 00-.4-1.5 1.59 1.59 0 00-1.28-.49 2 2 0 00-1.71.69 3.87 3.87 0 00-.53 2.3v4.31h-1.84v-8.64h1.44l.26 1.13h.06a2.53 2.53 0 011.11-.95 3.63 3.63 0 011.59-.34c2.08 0 3.11 1.06 3.11 3.17zM821.62 475.92a4.52 4.52 0 001.35-.21v1.38a3.6 3.6 0 01-.79.23 5.36 5.36 0 01-1 .09c-1.74 0-2.61-.92-2.61-2.75V470h-1.17v-.81l1.26-.68.63-1.82h1.13v1.92h2.46V470h-2.46v4.62a1.29 1.29 0 00.33 1 1.2 1.2 0 00.87.3zM824.59 466.32a1 1 0 112.06 0 1 1 0 01-.27.74 1.2 1.2 0 01-1.52 0 1 1 0 01-.27-.74zm1.94 10.93h-1.83v-8.64h1.83zM836.82 472.91a4.7 4.7 0 01-1.09 3.31 3.89 3.89 0 01-3 1.19 4.18 4.18 0 01-2.14-.55 3.63 3.63 0 01-1.43-1.57 5.41 5.41 0 01-.5-2.38 4.68 4.68 0 011.08-3.28 3.93 3.93 0 013-1.18 3.79 3.79 0 013 1.21 4.65 4.65 0 011.08 3.25zm-6.29 0c0 2 .74 3 2.21 3s2.19-1 2.19-3-.73-3-2.2-3a1.9 1.9 0 00-1.68.77 3.85 3.85 0 00-.52 2.23zM846.53 477.25h-1.84v-5.31a2.4 2.4 0 00-.4-1.5A1.59 1.59 0 00843 470a2 2 0 00-1.71.69 3.87 3.87 0 00-.53 2.3v4.31h-1.84v-8.64h1.44l.25 1.13h.1a2.53 2.53 0 011.11-.95 3.63 3.63 0 011.59-.34c2.08 0 3.11 1.06 3.11 3.17z"></path>
      </g>
      <g class="cls-1">
        <path class="cls-10" d="M723.07 164.65a3.32 3.32 0 01-1.17 2.74 5.15 5.15 0 01-3.34 1h-1.18v4.28h-1.87V161.2h3.29a5 5 0 013.2.87 3.14 3.14 0 011.07 2.58zm-5.69 2.12h1a3.51 3.51 0 002.11-.5 1.83 1.83 0 00.67-1.56 1.78 1.78 0 00-.6-1.47 3 3 0 00-1.88-.48h-1.29zM730.56 172.62l-.36-1.21h-.07a3.47 3.47 0 01-1.25 1.08 4 4 0 01-1.63.28 2.79 2.79 0 01-2-.68 2.61 2.61 0 01-.71-2 2.3 2.3 0 011-2 5.83 5.83 0 013-.74h1.49v-.46a1.72 1.72 0 00-.38-1.24 1.56 1.56 0 00-1.2-.41 4.11 4.11 0 00-1.27.19 11.11 11.11 0 00-1.18.46l-.59-1.31A6.7 6.7 0 01727 164a6.94 6.94 0 011.58-.19 3.74 3.74 0 012.49.72 2.85 2.85 0 01.84 2.26v5.82zm-2.73-1.25a2.26 2.26 0 001.6-.56 2 2 0 00.61-1.57v-.75h-1.11a3.81 3.81 0 00-1.93.51 1.33 1.33 0 00-.59 1.19 1.15 1.15 0 00.35.89 1.48 1.48 0 001.07.29zM742 172.62h-1.8v-5.32a2.32 2.32 0 00-.41-1.49 1.54 1.54 0 00-1.27-.49 2 2 0 00-1.71.69 3.77 3.77 0 00-.54 2.3v4.31h-1.83V164h1.44l.25 1.13h.1a2.53 2.53 0 011.11-1 3.63 3.63 0 011.59-.34c2.07 0 3.11 1.06 3.11 3.16zM747.51 172.77a3 3 0 01-2.51-1.17 6.44 6.44 0 010-6.6 3 3 0 012.54-1.18 3 3 0 012.59 1.26h.1a11 11 0 01-.14-1.47v-3.15h1.85v12.16h-1.44l-.32-1.14h-.09a2.93 2.93 0 01-2.58 1.29zm.49-1.48a2 2 0 001.65-.64 3.35 3.35 0 00.53-2.06v-.26a3.9 3.9 0 00-.53-2.32 2 2 0 00-1.67-.69 1.68 1.68 0 00-1.49.79 4 4 0 00-.52 2.23 4 4 0 00.51 2.2 1.71 1.71 0 001.52.75zM758.34 172.77a4.2 4.2 0 01-3.16-1.17 4.52 4.52 0 01-1.13-3.24 4.92 4.92 0 011.05-3.36 3.63 3.63 0 012.9-1.21 3.56 3.56 0 012.7 1 4 4 0 011 2.86v1h-5.76a2.76 2.76 0 00.68 1.93 2.32 2.32 0 001.8.68 7.15 7.15 0 001.43-.14 8.2 8.2 0 001.42-.48v1.49a6 6 0 01-1.36.45 8.3 8.3 0 01-1.57.19zm-.34-7.56a1.84 1.84 0 00-1.4.56 2.67 2.67 0 00-.63 1.61h3.92a2.41 2.41 0 00-.51-1.62 1.78 1.78 0 00-1.38-.55zM771.06 172.62h-1.84v-5.33a2.4 2.4 0 00-.38-1.48 1.37 1.37 0 00-1.17-.49 1.79 1.79 0 00-1.56.69 4.07 4.07 0 00-.49 2.3v4.31h-1.84V164h1.44l.26 1.13h.09a2.4 2.4 0 011-1 3.32 3.32 0 011.51-.34 2.62 2.62 0 012.64 1.36h.12a2.67 2.67 0 011.08-1 3.42 3.42 0 011.59-.36 2.9 2.9 0 012.26.78 3.47 3.47 0 01.71 2.38v5.64h-1.84v-5.33a2.4 2.4 0 00-.38-1.48 1.38 1.38 0 00-1.18-.49 1.8 1.8 0 00-1.56.67 3.44 3.44 0 00-.5 2zM779 161.69a1 1 0 112.06 0 .93.93 0 01-1 1 1 1 0 01-.77-.27 1 1 0 01-.29-.73zm1.94 10.93h-1.83V164h1.83zM787 172.77a3.8 3.8 0 01-3-1.14 5.8 5.8 0 01.05-6.64 4 4 0 013.08-1.17 5.76 5.76 0 012.46.51l-.55 1.47a5.63 5.63 0 00-1.92-.45c-1.5 0-2.25 1-2.25 3a3.55 3.55 0 00.56 2.18 1.93 1.93 0 001.64.73 4.65 4.65 0 002.32-.61v1.6a3.4 3.4 0 01-1 .42 6.6 6.6 0 01-1.39.1zM803.28 168.28a4.76 4.76 0 01-1.08 3.31 3.92 3.92 0 01-3 1.18 4.17 4.17 0 01-2.14-.54 3.63 3.63 0 01-1.43-1.57 5.44 5.44 0 01-.5-2.38 4.68 4.68 0 011.08-3.28 3.9 3.9 0 013-1.18 3.79 3.79 0 013 1.21 4.65 4.65 0 011.07 3.25zm-6.29 0c0 2 .74 3 2.21 3s2.19-1 2.19-3-.73-3-2.2-3a1.9 1.9 0 00-1.68.77 3.94 3.94 0 00-.51 2.23zM811.48 172.62l-.25-1.14h-.1a2.54 2.54 0 01-1.09 1 3.63 3.63 0 01-1.61.34 3.15 3.15 0 01-2.34-.78 3.22 3.22 0 01-.78-2.37V164h1.85v5.32a2.3 2.3 0 00.41 1.49 1.56 1.56 0 001.27.5 2 2 0 001.7-.69 3.75 3.75 0 00.55-2.32V164h1.84v8.64zM818.77 171.29a4.51 4.51 0 001.34-.21v1.38a3.65 3.65 0 01-.79.22 5.36 5.36 0 01-1 .09c-1.74 0-2.61-.91-2.61-2.75v-4.65h-1.18v-.82l1.26-.67.63-1.83h1.13V164H820v1.39h-2.46V170a1.29 1.29 0 00.33 1 1.25 1.25 0 00.9.29zM826.25 163.82a3 3 0 012.52 1.17 6.47 6.47 0 010 6.6 3.33 3.33 0 01-5.09 0h-.12l-.34 1h-1.37v-12.13h1.83v4.7h.08a2.91 2.91 0 012.49-1.34zm-.48 1.5a1.86 1.86 0 00-1.59.65 3.79 3.79 0 00-.51 2.19v.12a4.09 4.09 0 00.5 2.29 1.86 1.86 0 001.63.7 1.65 1.65 0 001.49-.77 4.16 4.16 0 00.5-2.23c0-1.97-.67-2.95-2.02-2.95zM836.19 163.82a4.51 4.51 0 01.91.08l-.18 1.71a3.59 3.59 0 00-.81-.09 2.4 2.4 0 00-1.79.71 2.64 2.64 0 00-.68 1.87v4.52h-1.84V164h1.44l.24 1.52h.1a3.41 3.41 0 011.12-1.23 2.68 2.68 0 011.49-.47zM842.48 172.77a4.2 4.2 0 01-3.16-1.17 4.52 4.52 0 01-1.13-3.24 4.92 4.92 0 011-3.33 3.65 3.65 0 012.9-1.21 3.56 3.56 0 012.7 1 4 4 0 011 2.86v1h-5.76a2.76 2.76 0 00.68 1.93 2.32 2.32 0 001.8.68 7.15 7.15 0 001.43-.14 8.2 8.2 0 001.42-.48v1.49a6 6 0 01-1.36.45 8.23 8.23 0 01-1.52.16zm-.34-7.56a1.84 1.84 0 00-1.4.56 2.67 2.67 0 00-.63 1.61H844a2.41 2.41 0 00-.51-1.62 1.78 1.78 0 00-1.35-.55zM853.33 172.62l-.37-1.21h-.06a3.57 3.57 0 01-1.26 1.08 4 4 0 01-1.62.28 2.78 2.78 0 01-2-.68 2.57 2.57 0 01-.72-2 2.3 2.3 0 011-2 5.87 5.87 0 013-.74h1.49v-.46a1.72 1.72 0 00-.39-1.24 1.53 1.53 0 00-1.2-.41 4.16 4.16 0 00-1.27.19 10.17 10.17 0 00-1.17.46l-.59-1.31a6.61 6.61 0 011.53-.56 7 7 0 011.58-.19 3.74 3.74 0 012.49.72 2.81 2.81 0 01.84 2.26v5.82zm-2.74-1.25a2.29 2.29 0 001.61-.56 2 2 0 00.6-1.57v-.75h-1.1a3.75 3.75 0 00-1.89.43 1.33 1.33 0 00-.59 1.19 1.15 1.15 0 00.34.89 1.49 1.49 0 001.03.37zM859 168.05l1-1.29 2.6-2.78h2.12l-3.47 3.71 3.69 4.93h-2.14l-2.8-3.79-1 .83v3h-1.8v-12.2h1.8v5.93l-.09 1.66zM866.08 165a1 1 0 011.15-1.18 1.14 1.14 0 01.88.31 1.45 1.45 0 010 1.75 1.15 1.15 0 01-.87.32 1.07 1.07 0 01-.85-.32 1.24 1.24 0 01-.31-.88zm0 6.67a1.18 1.18 0 01.3-.87 1.1 1.1 0 01.85-.31 1.15 1.15 0 01.87.32 1.2 1.2 0 01.3.86 1.25 1.25 0 01-.3.88 1.11 1.11 0 01-.87.32 1.07 1.07 0 01-.85-.32 1.22 1.22 0 01-.3-.91zM722.91 191.82h-1.84v-5.31a2.34 2.34 0 00-.4-1.5 1.59 1.59 0 00-1.28-.49 2 2 0 00-1.7.69 3.79 3.79 0 00-.54 2.3v4.31h-1.84v-8.64h1.44l.26 1.13h.09a2.53 2.53 0 011.11-.95 3.63 3.63 0 011.59-.34c2.08 0 3.11 1.06 3.11 3.17zM733.13 187.48a4.76 4.76 0 01-1.08 3.31 3.9 3.9 0 01-3 1.18 4.2 4.2 0 01-2.14-.54 3.69 3.69 0 01-1.43-1.57 5.44 5.44 0 01-.5-2.38 4.68 4.68 0 011.02-3.28 3.9 3.9 0 013-1.18 3.79 3.79 0 013 1.21 4.65 4.65 0 011.13 3.25zm-6.29 0c0 2 .74 3 2.21 3s2.19-1 2.19-3-.73-3-2.2-3a1.9 1.9 0 00-1.68.77 3.94 3.94 0 00-.52 2.23zM739.29 180.89a1 1 0 01.27-.76 1 1 0 01.77-.26 1 1 0 01.75.26 1 1 0 01.27.76.93.93 0 01-1 1 1 1 0 01-.77-.27 1 1 0 01-.29-.73zm1.94 10.93h-1.83v-8.64h1.83zM751.46 191.82h-1.84v-5.31a2.33 2.33 0 00-.41-1.5 1.56 1.56 0 00-1.27-.49 2 2 0 00-1.71.69 3.87 3.87 0 00-.53 2.3v4.31h-1.84v-8.64h1.44l.25 1.13h.1a2.53 2.53 0 011.11-.95 3.63 3.63 0 011.59-.34q3.1 0 3.11 3.17zM757.23 190.49a4.52 4.52 0 001.35-.21v1.38a3.65 3.65 0 01-.79.22 5.36 5.36 0 01-1 .09c-1.74 0-2.61-.91-2.61-2.75v-4.65H753v-.81l1.26-.68.63-1.82H756v1.92h2.46v1.39H756v4.62a1.3 1.3 0 00.33 1 1.23 1.23 0 00.9.3zM764.08 192a4.16 4.16 0 01-3.15-1.17 4.46 4.46 0 01-1.14-3.24 4.92 4.92 0 011.05-3.33 3.65 3.65 0 012.9-1.21 3.6 3.6 0 012.71 1 4 4 0 011 2.86v1h-5.76a2.88 2.88 0 00.68 1.94 2.41 2.41 0 001.8.67 7.15 7.15 0 001.43-.14 8.2 8.2 0 001.42-.48v1.49a5.86 5.86 0 01-1.36.45 8.23 8.23 0 01-1.58.16zm-.34-7.56a1.84 1.84 0 00-1.4.56 2.67 2.67 0 00-.63 1.61h3.92a2.41 2.41 0 00-.51-1.62 1.8 1.8 0 00-1.38-.58zM773.91 183a4.51 4.51 0 01.91.08l-.18 1.71a3.59 3.59 0 00-.81-.09 2.36 2.36 0 00-1.79.72 2.6 2.6 0 00-.68 1.86v4.52h-1.84v-8.64H771l.24 1.52h.1a3.49 3.49 0 011.12-1.23 2.68 2.68 0 011.45-.45zM778.39 191.82l-3.28-8.64h1.94l1.75 5a9.34 9.34 0 01.55 2.05h.06a13.85 13.85 0 01.55-2.05l1.76-5h1.95l-3.29 8.64zM788.76 192a4.16 4.16 0 01-3.15-1.17 4.46 4.46 0 01-1.14-3.24 4.92 4.92 0 011-3.33 3.65 3.65 0 012.9-1.21 3.56 3.56 0 012.7 1 3.94 3.94 0 011 2.86v1h-5.76a2.88 2.88 0 00.68 1.94 2.41 2.41 0 001.8.67 7.15 7.15 0 001.43-.14 8.2 8.2 0 001.42-.48v1.49a5.86 5.86 0 01-1.36.45 8.23 8.23 0 01-1.52.16zm-.34-7.56a1.84 1.84 0 00-1.4.56 2.67 2.67 0 00-.63 1.61h3.92a2.41 2.41 0 00-.51-1.62 1.81 1.81 0 00-1.38-.58zM801.8 191.82H800v-5.31a2.34 2.34 0 00-.4-1.5 1.59 1.59 0 00-1.28-.49 2 2 0 00-1.7.69 3.79 3.79 0 00-.54 2.3v4.31h-1.88v-8.64h1.44l.26 1.13h.1a2.53 2.53 0 011.11-.95 3.64 3.64 0 011.6-.34q3.1 0 3.1 3.17zM807.58 190.49a4.46 4.46 0 001.34-.21v1.38a3.54 3.54 0 01-.78.22 5.47 5.47 0 01-1 .09c-1.74 0-2.61-.91-2.61-2.75v-4.65h-1.18v-.81l1.27-.68.62-1.82h1.14v1.92h2.46v1.39h-2.46v4.62a1.3 1.3 0 00.33 1 1.21 1.21 0 00.87.3zM810.54 180.89a1 1 0 01.27-.76 1 1 0 01.77-.26 1 1 0 01.75.26 1 1 0 01.27.76.93.93 0 01-1 1 1 1 0 01-.77-.27 1 1 0 01-.29-.73zm1.94 10.93h-1.83v-8.64h1.83zM822.77 187.48a4.76 4.76 0 01-1.08 3.31 3.9 3.9 0 01-3 1.18 4.2 4.2 0 01-2.14-.54 3.69 3.69 0 01-1.43-1.57 5.44 5.44 0 01-.5-2.38 4.68 4.68 0 011.08-3.28 3.9 3.9 0 013-1.18 3.79 3.79 0 013 1.21 4.65 4.65 0 011.07 3.25zm-6.29 0c0 2 .74 3 2.22 3s2.18-1 2.18-3-.73-3-2.2-3a1.9 1.9 0 00-1.68.77 3.94 3.94 0 00-.52 2.23zM832.48 191.82h-1.84v-5.31a2.34 2.34 0 00-.4-1.5 1.59 1.59 0 00-1.28-.49 2 2 0 00-1.7.69 3.79 3.79 0 00-.54 2.3v4.31h-1.84v-8.64h1.44l.26 1.13h.09a2.53 2.53 0 011.11-.95 3.64 3.64 0 011.6-.34q3.11 0 3.1 3.17z"></path>
      </g>
      <path stroke="#677279" stroke-miterlimit="10" fill="none" d="M703.3 168.37H557.07"></path>
      <circle class="cls-9" cx="460" cy="168.37" r="4"></circle>
      <path stroke-miterlimit="10" stroke="#fff" fill="none" d="M557.07 168.37H460"></path>
    </g>
  </g>
</svg>

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
eyJoaXN0b3J5IjpbMTI5MzkwMzM4MywtODkwNDU2OTAsLTgwMz
M1MTE5MCwtOTgxMTUwMzAsLTIwMDQ5NDg1OTEsMTYwODc2ODU2
OCwxMjY5MDU1NDgwLDEyMTU4MDU4ODgsLTE5NjIyNDc1MTcsLT
E3MTcxMDUzNTZdfQ==
-->