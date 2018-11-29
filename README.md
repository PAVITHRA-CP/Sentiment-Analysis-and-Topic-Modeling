# Sentiment-Analysis-and-Topic-Modeling

Extract comments from glassdoor reviews, performing polarity analysis and topic modeling for dataset. In this work, glassdoor review dataset is used. The dataset contains different fields like date, pros, cons etc.. Change in intensity of comments for every quarter of year is calulated from the dataset and for this fields topic is also identified.

The solution is split into 4 task :

      (1) Data – Cleaning
      (2) Sort the dataset quarterly
      (3) Polarity Analysis
      (4) Topic Modeling

# Data Cleaning
The dataset contains characters other than unicode. The data is cleaned by reading the
characters as unicode and ignoring the non-unicode characters.

# Sort the dataset quarterly
The dataset is sorted on the basis on date field. For each sorted date, the quarter to which the
date belongs is identified and this value is added to as a new column in the dataset.

# Polarity Analysis
On each quarter wise sorted data, the polarity of pros and cons for each quarter is identified.
From this output the change in intensity of the comments can be calculated.

# Topic Modeling
The topic modeling is done on the fields title, pros and cons. Each field were taken
seperately and the topic set for each field is created. Each field is preprocessed by tokenization and
then using this tokens LDA modeling is performed for topic identification. LDA is performed using
Genism. The topic set is set to 5 and each set contains 6 words.
