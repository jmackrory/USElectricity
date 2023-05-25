# Exploring the US Electrical Grid and Demand Forecasting

## Project Summary

This project aims to explore the US electrical grid using open data from the
Energy Information Agency (www.eia.gov/opendata), and to develop models predicting demand for electricity.
This project includes gathering data, exploring the broad trends of the US electrical grid, and
then developing models to forecast day-ahead electricity demand.

One of the key tasks for a power company is to forecast electrical demand for the next day, and ensure
that adequate electricity is generated to meet that demand.  Overestimating demand wastes energy,
while underestimating leads to expensive purchases from other companies, or running expensive
generators to stave off brownouts.

This project has two phases: one exploring electricity power generation in the US since 2001,
and trying to forecast electricity demand based on two years worth of hourly time data, and weather
data.
The project will use Python and Pandas to explore the data, along with a little SQL, as the data are too large to fit into Pandas dataframes.
I have tried to develop a seasonal exponential smoothing model  (EBA\_seasonal.ipynb) for electricity demand, as well
as a recurrent neural network model (EBA\_RNN.ipynb)
Currently, this work focuses just on Portland, but could be generalized to include other locations.
Both models take the temperature, and prior demand, and attempt to forecast for the next 24 hour
period.

## Sources of Data

The electricity datasets were downloaded as bulk files from https://www.eia.gov/electricity/data.php.
The real-time electricity data (EBA.txt) was downloaded from the EIA in October 2017,
alongside a more general dataset on the grid (ELEC.txt) from January 2017.

ELEC.txt has state and plant level statistics, with monthly and quarterly resolution.
The data include total generation across the US for all sources, quality of fuels, price, and others.
This data will be used to get a summary of the grid, and look at some longer term and geographic trends.

EBA.txt has 2 years of data showing with hourly time resolution, showing electricity generation, demand,
demand forecasts, and interchange (sales between companies).
This is presented for all independent system operators (ISO).
This data will be used to generate my own demand forecasts.

The weather also plays a large role in how much electricity is used.
The National Climactic Data Center (NCDC) stores a variety of historical weather records.
I will use the Integrated Surface Database, which contains simplified hourly observations on temperature,
rainfall, and wind from automated weather stations located at airports.
The weather data was downloaded from NOAA's FTP server (ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/).
I have found a list of the three biggest cities in each state, and matched up the airport ID in each
city, with the desired station IDs, and downloaded the data for 2015,2016,2017 for each station.


## Files Contained:

This repo includes the following files:
Split roughly according to acquiring data, loading data, exploring data, and building models.

### Acquiring/loading data utilities:
  - get\_weather\_data.py - match airport codes for largest cities with weather station IDs, and download data.
  - weather\_dataframe.py - read in weather data for particular city, state, and convert to pandas dataframe
  - json\_to\_sql.py - code to load ELEC file into a SQL database for easier access.
                    SQL was essential for being able to explore the data at all on my laptop.

### Exploring data notebooks:
  - ELEC\_explore.ipynb  - load ELEC data from SQL database, makes plots of energy mix on regional level.
                       Also some exploration of EBA data.
  - EBA\_explore.ipynb - load EBA and weather data, explore, play with patterns, look at data quality.

### Demand Model notebooks:
  - EBA\_seasonal.ipynb - build simple seasonal models account for temperature to predict demand.
  - EBA\_RNN.ipynb - build a recurrent neural network (RNN) on temperature, and demand data.
  - EBA\_fft.py - functions to try filtering seasonal patterns from FFT of EBA data for use with EBA\_seasonal.ipynb.

### Libraries:
 - EBA\_RNN.py - code for making a RNN in EBA\_RNN.ipynb
 - EBA_seasonal/EBA\_multiseasonal.py - code for making a exponential smoothing model.
 - EBA_seasonal/EBA\_multiseasonal\_temp.py - extends multiseasonal to include temperature.
 -


### Running Notes
As of 2023, have migrated to using Docker particularly for using Tensorflow.
Currently working to refactor library code into package to allow install with clearer simple scripts.  This should allow easier reproducibility, and portability for other projects.

Dev environment is a modified Tensorflow/Jupyter docker container interacting with a PostgreSQL container.  Jupyter server runs inside container and is available for interactive running / debugging.