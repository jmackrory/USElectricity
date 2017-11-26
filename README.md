# US-Electricity
============================================================

## Goal

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

My other goal of trying to predict solar generation can be explored via data provided by 
California Distributed Generation Statistics. This was a small scale solar program that output 
15 minute time resolution over a few years.  I intend to develop a model to predict output 
for each cell by modelling against satellite data.  This is essentially a separate project. 

The project will use Python and Pandas to explore the data.
Initially however I will stick to plotting, summary statistics, linear regression,
and Fourier analysis. Obviously, electricity usage is strongly seasonal, and has daily oscillations.
Ideally, if the data supports it, I would like to try applying some machine learning techniques
to it.  I plan to use TensorFlow to train a recurrent neural network in both cases.
I will try to build a simple model based on historical weather data, and time of year. 

## Exploratory Questions
* What is the mix of energy used in electricity generation across the US?
* How has that mix evolved over time?
* What is the cost of generation for each fuel type?
* How has renewable energy use grown in the US over time, and geographically?
* How does demand vary over the day, monthly and yearly timescale?
* What sectors are using the electricity (commercial, industrial, residential, agricultural)?
* How does renewable energy generation correlate with the weather data that is available?

## Timeline and Plan
This is an attempt to plan out what I'm doing.
1. By the end of the week make a plots of US energy mix as a whole over time.
2. Also plot dominant energy source for each state.
3. 

# Prediction Goals
*Can I predict the renewable energy power share based on the weather forecast?
*Can I predict the demand for electricity a day ahead?
*Can I predict the price of electricity contracts a day ahead?

### Changing Focus
The initial attempt with the Bulk data lead to getting stuck trying to manipulate
enormous data files. Using SQL alongside chunking the files made it possible to get at
the state level summaries I wanted.  
Perhaps then going back to comparing demand forecasts for machine learning.

## Sources of Data

The electricity data has been taken from EIA.gov in February, for the real-time grid data (EBA.txt),
as well as broad summary data on electricity (ELEC.txt) from 
https://www.eia.gov/electricity/data.php

ELEC.txt and EBA.txt were acquired from EIA.gov on January 21, 2017,
via their opendata portal.
ELEC.txt has summary and plant level statistics of generation across the
US for types of generation, as well as quality of fuels.
EBA.txt has around 1 year of data showing in hour chunks the
net generation across the US, at an interconnect level, alongside demand forecasts.  

I was hoping to do more studies on the temporal variability at a local level of
solar and wind sources.  
The National Renewable Energy Laboratory (NREL) has a great deal of data 
on output from existing solar and wind plants, as well as projections
for where plants could be installed.  

Another source is the California Distributed Generation Statistics
(http://www.californiadgstats.ca.gov), which mostly covers solar generation.
This includes numerous small scale projects over a couple years. 
The project names are tied to locations for applications in CSI Working DataSet.
The actual datafiles have a row for each station, with 15 min time resolution for a few years. 

Weather data is available from NCDC.
I have found a list of biggest cities in each state.
I will use the Integrated Surface Database, which contains simplified hourly observations. 
These will typically have airports with weather stations.  I aim to find the airports call signs,
then find their station IDs.  The weather data can be downloaded from 
NOAA's FTP server (ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/). 
The files seem modest (100KB/station/year). 

Some satellite imagery might be available from CLASS.  This is rather large to deal with.  
I am trying to find a way to get just the relevant subsets over the cities for which I have 
solar electricity data, namely San Diego and Los Angeles from 2010-2016.

### Prior research on Solar Forecasting

I've found research that already does this, particularly from Carlos Coimbra's group as UC, San Diego.
This level of work seems to be a PhD project on its own. 
(Using neural networks with previous data as well as local sensors, and forecasts to predict solar output,
for a single plant.)

### Extracting Data
I used awk to pull out the names and series ID from the bulk data.
To extract series_id, and names.
"cat ELEC.txt | awk --field-separator=, '{print $1, $2}' ELEC_id_name.txt"

For a general picture of US electricity generation, I think I want the ELEC.GEN
ELEC.PLANT is too finegrained for this initial survey.

### Putting into SQL
I split the large-ish data files into chunks of 1000 lines using:
"split --lines=10000 -d ELEC.txt split_dat/ELEC"
(The splitting was necessary since my laptop ran out of memory trying to read
in the whole JSON file).

Those sub-chunks were then read into Python with Pandas.
Appended the data frames to a SQL database for easier sectioning.
(This is carried out in json_to_sql.py)

Dropped columns:


