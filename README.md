# US-Electricity
============================================================

## Project Summary

This project aims to explore the US electrical grid using open data from the 
Energy Information Agency (www.eia.gov/opendata), and to develop models predicting demand for electricity.
This project includes gathering data, exploring the broad trends of the US electrical grid, and 
then developing models to forecast day-ahead electricty demand.

One of the key tasks for a power company is to forecast electrical demand for the next day, and ensure
that adequate electricity is generated to meet that demand.  Overestimating demand wastes energy, 
while underestimating leads to expensive purchases from other companies, or running expensive
generators to stave off brownouts.

The project will use Python and Pandas to explore the data.  I used SQL to handle some of the 
datasets, as they were too large to fit into Pandas dataframes.
I have tried to develop a seasonal exponential smoothing model  (EBA_seasonal.ipynb) for electricity demand, as well
as a recurrent neural network model (EBA_RNN.ipynb)
Currently, this work focuses just on Portland, but could be generalized to include other locations.
Both models take the temperature, and prior demand, and attempt to forecast for the next 24 hour 
period. 

## Sources of Data

The electricity datasets were downloaded as bulk files from https://www.eia.gov/electricity/data.php.
(An API is available, but for a newcomer, the complicated series names made extracting and exploring the data
very difficult.  I used a SQL database to then manage the Bulk data, which was too large to fit in memory)
The real-time electricity data (EBA.txt) was downloaded from the EIA in October 2017,
alongside a more general dataset on the grid (ELEC.txt) from in January 2017. 

ELEC.txt has summary and plant level statistics, with monthly and quarterly resolution.
The data include total generation across the US for all sources, quality of fuels, price, and others.
This data will be used to get a summary of the grid, and look at some long term and geographic trends.

EBA.txt has 2 years of data showing with hourly time resolution, showing electricity generation, demand,
demand forecasts, and interchange (sales between companies).  
This is presented for all independent system operators (ISO).  
This data will be used to generate my own demand forecasts. 

The weather also plays a large role in how much electricity is used. 
The National Climactic Data Center (NCDC) stores a variety of historical weather records.
I will use the Integrated Surface Database, which contains simplified hourly observations on temperature,
rainfall, and wind from automated weather stations located at airports.  
The weather data can be downloaded from NOAA's FTP server (ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/). 
I have found a list of the three biggest cities in each state, and matched up the airport IDS in each
city, with the desired station IDs, and downloaded the data for 2015,2016,2017 for each station.
The files seem modest (100KB/station/year). 

## Files Contained:

This repo includes the following files: 

Split roughly according to acquiring data, loading data, exploring data, and bulding models.

*Acquiring/loading data:
 - get_weather_data.py - match airport codes for largest cities with weather station IDs, and download data.
 - weather_dataframe.py - read in weather data for particular city, state, and convert to pandas dataframe
 - json_to_sql.py - code to load ELEC file into a SQL database for easier access.  
                    SQL was essential for being able to explore the data at all on my laptop. 

* Exploring data:
 - SQLexplore.ipynb  - load ELEC data from SQL database, makes plots of energy mix on regional level. 
                       Also some exploration of EBA data.
 - EBA_explore.ipynb - load EBA and weather data, explore, play with patterns, look at data quality.
 
* Demand Models: 
 - EBA_seasonal.ipynb - build simple seasonal models account for temperature to predict demand.
 - EBA_fft.py - functions to try filtering seasonal patterns from FFT of EBA data for use with EBA_seasonal.ipynb.
 - EBA_RNN.ipynb - build a recurrent neural network (RNN) on temperature, and demand data. 

* Should be removed: (mostly a product of obsolete files that should have been removed from analysis)
- load_data.py (from earlier attempts at trying to split ELEC at state level via grep)
- sqlexplore.py - superceded by SQL_explore.ipynb
- test_sql.py - testing/play functions to check I can load/reload from SQL.
- demand_rnn.py - initial attempt at coding RNN, was going to use primitive tensorflow.  Never finished.
- Todo.md - super out of date, pointless

## Exploratory Questions
* What is the mix of energy used in electricity generation across the US?
* How has that mix evolved over time?
* What is the cost of generation for each fuel type?
* How has renewable energy use grown in the US over time, and geographically?
* How does demand vary over the day, monthly and yearly timescale?
* What sectors are using the electricity (commercial, industrial, residential, agricultural)?
* How does renewable energy generation correlate with the weather data that is available?

# Prediction Goals
* Can I predict the demand for electricity a day ahead?
* Can I predict the price of electricity contracts a day ahead?

## Further projects

The electrical grid is modernizing to include more renewable sources (such as wind and solar energy),
driven by economies of scale with the new technologies, and the need to reduce CO2 emissions to limit the extent of climate change.
These renewable energy sources have considerable variability throughout the day, and there is a need
to be able to accurately forecast generation from these sources.  At some point I would like to tackle this
more advanced forecasting task.

My other goal of trying to predict solar generation can be explored via data provided by 
California Distributed Generation Statistics. This was a small scale solar program that output 
15 minute time resolution over a few years.  I intend to develop a model to predict output 
for each cell by modelling against satellite data.  This is essentially a separate project. 

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

Some satellite imagery might be available from CLASS.  This is rather large to deal with (100MB per image).  
I am trying to find a way to get just the relevant subsets over the cities for which I have 
solar electricity data, namely San Diego and Los Angeles from 2010-2016.

### Prior research on Solar Forecasting

I've found research that already does this, particularly from Carlos Coimbra's group as UC, San Diego.
This level of work seems to be a PhD project on its own. 
(Using neural networks with previous data as well as local sensors, and forecasts to predict solar output,
for a single plant.)


