# US-Electricity

#Goal

This project aims to download and explore data about the US
electrical grid and energy system based on data from the US Energy Information agency at
www.eia.gov/opendata. This system has data on a state-county-plant level, at temporal resolution
of months/quarters in most cases.  In some cases it may be possible to explore near realtime data
about power generation and demand.  
In particular I want to explore the mix of fuel types used,
how/when is electricity used, and the cost of buying it.  The notions tie into applying
data science to develop the smart grid. Can we predict demand and supply of electricity
from renewable sources?  I plan to compare the renewable energy data with the
available weather data from NOAA to build a model to predict 

The project will use Python and Pandas to explore the data.
Initially however I will stick to plotting, summary statistics, linear regression,
and Fourier analysis. Evidently electricity usage is strongly seasonal, and has daily oscillations.
Ideally, if the data supports it, I would like to try applying some machine learning techniques
to it.

#Exploratory Questions
* What is the mix of energy used in electricity generation across the US?
* How has that mix evolved over time?
* What is the cost of generation for each fuel type?
* How has renewable energy use grown in the US over time, and geographically?
* How does demand vary over the day, monthly and yearly timescale?
* How does renewable energy correlate with the weather?

#Prediction Goals
*Can I predict the renewable energy power share based on the weather forecast?
*Can I predict the demand for electricity a day ahead?
*Can I predict the price of electricity contracts a day ahead?


#Changing Focus
The initial attempt with the Bulk data lead to getting stuck trying to manipulate
enormous data files. I will instead try to use the API to get the data I am most interested
in for this initial analysis - state level data at the monthly level for many years.
Perhaps then going back to comparing demand forecasts for machine learning.
