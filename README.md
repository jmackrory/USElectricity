# US-Electricity
============================================================

##Goal

This project aims to download and explore data about the US
electrical grid and energy system based on data from the US Energy Information agency at
www.eia.gov/opendata. This system has data on a state-county-plant level, at temporal resolution
of months/quarters in most cases.  In some cases it may be possible to explore near realtime data
about power generation and demand.
In particular I want to explore the mix of fuel types used,
how/when is electricity used, and the cost of buying it.  The notions tie into applying
data science to develop the smart grid. Can we predict demand and supply of electricity
from renewable sources?  

I initially hoped to explore predictions about the type of generation available during the day, with
fine resolution.  However, that data is not available in the EIA realtime dataset.
What is available is the net generation by power supplier, and the net interchange.
This could be studied with a similar goal: predict the net demand a day ahead, and the interchange.

(Such finegrained data is not actually available in this dataset - what is available is
net generation, and interchange between power companies.) 

The project will use Python and Pandas to explore the data.
Initially however I will stick to plotting, summary statistics, linear regression,
and Fourier analysis. Obviously, electricity usage is strongly seasonal, and has daily oscillations.
Ideally, if the data supports it, I would like to try applying some machine learning techniques
to it.
I will try to build a simple model based on historical weather data, and time of year. 

## Data Sources


The electricity data has been taken from EIA.gov in February, for the real-time grid data (EBA.txt),
as well as broad summary data on electricity (ELEC.txt) from 
https://www.eia.gov/electricity/data.php

The weather data is proving more difficult to find.  Weather Underground has hourly data
for all cities for at leas the previous few years (but it's not free). 
The Local Climatological Data from NOAA's Climate Data Online service seems to fit the bill. 
(https://www.ncdc.noaa.gov/cdo-web/datasets)
Can get city data, with hourly resolution in monthly chunks.





##Exploratory Questions
* What is the mix of energy used in electricity generation across the US?
* How has that mix evolved over time?
* What is the cost of generation for each fuel type?
* How has renewable energy use grown in the US over time, and geographically?
* How does demand vary over the day, monthly and yearly timescale?
* What sectors are using the electricity (commercial, industrial, residential, agricultural)
* How does renewable energy generation correlate with the weather data that is available?

##Timeline and Plan
This is an attempt to plan out what I'm doing.
1. By the end of the week make a plots of US energy mix as a whole over time.
2. Also plot dominant energy source for each state.
3. 


#Prediction Goals
*Can I predict the renewable energy power share based on the weather forecast?
*Can I predict the demand for electricity a day ahead?
*Can I predict the price of electricity contracts a day ahead?

###Changing Focus
The initial attempt with the Bulk data lead to getting stuck trying to manipulate
enormous data files. I will instead try to use the API to get the data I am most interested
in for this initial analysis - state level data at the monthly level for many years.
Perhaps then going back to comparing demand forecasts for machine learning.

###Sources of Data
ELEC.txt and EBA.txt were acquired from EIA.gov on January 21, 2017,
via their opendata.
ELEC.txt has summary and plant level statistics of generation across the
US for types of generation, as well as quality of fuels.
EBA.txt has around 1 year of data showing in hour chunks the
net generation across the US, alongside demand forecasts.  


###Extracting Data
I used awk to pull out the names and series ID from the bulk data.
To extract series_id, and names.
"cat ELEC.txt | awk --field-separator=, '{print $1, $2}' ELEC_id_name.txt"

For a general picture of US electricity generation, I think I want the ELEC.GEN
ELEC.PLANT is too finegrained for this initial survey.

###Putting into SQL
Split the large-ish data files into chunks of 1000 lines using:
"split --lines=10000 -d ELEC.txt split_dat/ELEC"
(The splitting was necessary since my laptop ran out of memory trying to read
in the whole JSON file).

Then read those into Python with Pandas.
Appended the data frames to a SQL database for easier sectioning.

Dropped columns:
