## Steps needed to run the analysis code on another computer: 

The code here within this project takes both environmental and biological parameters to run 3 separate PCAs (biological, environmental, and combined environmentla and biological parameters). This could be used for other datasets, but the indication of which columns pertain to biological or environmental data would need to be altered within the code. (Note: these PCAs can only be run win quantitative and ordinal data, thus any qualitative or non linear scaling numbers should be avoided if possible). In addition all of the packages imported at the beginning of the code should be installed and working on your computer. The maptools.py file must also be present in order to run this program, and should be included within the respository for this project. At the end of this code there are lots of exploratory plots to graphically represent possible important factors as they relate to different isotope values. This would require data for length, weight, sexual maturity, gender, delta 13C, delta 15N, C:N, and management region data, but could be edited to fit individual data sets or researcher needs.

## Additional packages 

This code also utilizes the maptools.py package to create a map of my sampling ports and fishery management locations. The goal of this package is to return axis object for cartopy map with labeled gridlines With inputs: Cartopy projection, default = cartopy.crs.PlateCarree() and the output: Cartopy axis object. This package is based on blog post by Filipe Fernandes: https://ocefpaf.github.io/python4oceanographers/blog/2015/06/22/osm/ License: Creative Commons Attribution-ShareAlike 4.0 https://creativecommons.org/licenses/by-sa/4.0/.
    
## Location/source of data: 

This data is derived from my thesis data which originally came from Rachel Brooks' Canary Rockfish samples for her life history thesis (Moss Landing Marine Laboratories Ichthyology Lab. These fish samples were taken over the course of 2 years from 2017-2018 at 13 different ports across Washington, Oregon, and California. 

## Variables used within data:
TL cm: Total Length in centimeters of individual fish samples. Thus the measurement of the entire length of the fish from mouth to terminus. I selected to use this measurement over standard or fork length as they are the same species thus the morphology shouldn't differ drastically enough to affect total, standard, or fork lengths 

Fulton's K: This is a measurement = 100*(weight/length^3) which is essentially a measurement of proportionality between length and weight and can be used as a proxy for if a fish if over or under standarderized weights. A value ~1 means the fish is likely a healthy/normal ratio between length and weight. Values>1 indicates a fatter fish realtive to length and values<1 indicate a skinny fish.

GSI (g): gonadosomatic index is a measure of gonad size:fish size. Can be used as an indicator of gonadal development in fish thus their sexual maturity. Indicates the percentage of body weight being utilized for production for sexual constituents (ie sperm or eggs)

HSI (g): Hepatosomatic index indicates liver weight:total body weight. This is used as a proxy for the measurement of energy reserves of an animal.

Weight: Total weight in grams of fish samples 

mgmt_lat: There are only 5 values for this parameter which pertain to a single latitude measurement at the end of each of the 5 fishery management zones along the CCS. Vancouver (47.3), Columbia(43.00), Eureka(40.10), Monterey (36.00), and Conception (32.00)

Depth m: Depth at which each individual samples was fished from water column 

Lat: Latitude where fish was caught

Long: Longtiude where fish was caught

Relief: Measured on a scale of 0,1,2,3 and indication of habitat complexity in which fish was caught. Reloefe refers to the topographic variation across a landscape with 0 being a flat plain and 3 being a very hilly and highly variable topography realtive to elevation.

Maturity: denoted either by IMM or MAT which stands for immature or mature respectively. There are a few variables denoted by UNK.

Sex: denoted by F or M which stands for female or male respectively 

d13C: delta 13Carbon which is the ratio of 13C:12C isotopes present in the sample. The more positive the value indicates an enrichment of the heavier isotope, 13C. A more negative value indicates a depletion in 13C. This is used as a proxy for primary productivty at the base of the respective foodweb as carbon values don't fractionate heaviliy throughout trophic levels 

d15N: delat 15Nitrogen which is the ratio of 15N:14N present in the sample. The more positive the value indicates an enrichment of the heavier isotope, 15N. A more negative value indicates a depletion in 15N. This is used as a proxy for where the organism is eating on the trophic level as nitrogen fractionates 3-4% per trophic level.

C:N: The ratio of delta 13C values to delta 15N values, used as a proxy for trophic position.

## Location of data in repository/how to access data: 

The original data sheet was formatted specifically for this project as there are many other unecessary variables included in the master sheet concerning non-relevant material. Thus, the datasheet utilized in this project can be found within the github repository as 'pythonproject.csv'