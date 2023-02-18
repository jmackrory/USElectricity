/*Create a base schema common for all ISOs
id - Actual Net Interchange
d - demand
ti - total interchange
ng - Net Generation

ng sub varieties:
ng - Natural Gas
wat - Hydro
sun - Solar
col - Coal
wnd - Wind
oth - other (biomass etc
oil - Oil
nuc - Nuclear

- hl local time
- h UTC

*/

CREATE SCHEMA IF NOT EXISTS iso
       CREATE TABLE id_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE id_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE d_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE d_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ti_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ti_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE df_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE df_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_ng_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_ng_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wat_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wat_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_sun_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_sun_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_col_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_col_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oth_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oth_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wnd_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wnd_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oil_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oil_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_nuc_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_nuc_hl (dt TIMESTAMP WITH TIME ZONE, val real)
