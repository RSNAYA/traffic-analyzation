
# Traffic Analyzer

This code will process all the images in the specified directories, extract data on the traffic levels in the parking lot, and analyze the data to understand how different conditions (e.g., time of day, presence of events) are associated with traffic levels.

The code first groups the data by condition and calculates the mean traffic level for each group. It then plots the mean traffic levels by condition using a bar chart, which provides a visual representation of the data.

Finally, the code calculates the correlation between traffic levels and conditions, which can help to understand the strength and direction of the relationship between these variables. The correlation is calculated using the corr() function from the pandas library, which returns a correlation matrix containing the Pearson correlation coefficient (r) for each pair of variables.
