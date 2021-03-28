# Birding Data Analysis - Senior Project
Please note that this is not a finalized repository! I am currently refactoring the code to make it cleaner and more efficient.

Hello! My name is Conner Sucic and this was my senior capstone project for the Bachelor in Applied Computer Science from Birmingham-Southern College.

For my project, I found the relationship between drought levels and bird sightings of warblers in the Southeastern United States. My drought data was taken from the United States Drought Monitor and my bird sighting data was from the website eBird.

## eBird Data Preprocessing

The eBird data was downloaded from their website as a .txt file. The raw bird sighting data I downloaded, for all sightings before 2020, was 187GB. This file was too large for Python to deal with as a whole, but luckily eBird has their own R library called Auk that filters the dataset based on user conditions. Using Auk, I pulled datasets from the raw bird sighting data file with the following conditions:

* All warblers were pulled (36 species)
* Years 2002-2019 were pulled
* Only sightings in the Southeastern United States (8 states) 

After this preprocessing, there were 18 .txt files of bird sighting data, one for each year.
