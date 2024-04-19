# BME_repetitions
Calculates repetition for a conversation file structured with the BME method


 - This code looks at an INPUT dataframe constituted of turns from 3 different speakers (referred as MOD, P1, P2), 
 each being displayed in a different column according to the BME method (sample available, see file "input_example.csv").
 - The output of this code is an OUTPUT dataframe similar to the input's but augmented with 
 several columns including the number of items available and repeated for each turn, along with the final Jaccard Index.

 The end goal of this code is:
 - From one turn to another, it counts repetitions as a Jaccard Index (i.e. how similar are two turns?)
 It accounts for several types of repetitions: Self-Repetitions, Other-Repetitions.
 It accounts for three natures of repeated items: Open-Class Items, Closed-Class Items, Both undistinctively (also call "all").
 It does the calculation for several n-grams.

## To test this code, a sample of the Multisimo Corpus (first 20 seconds of every conversation is provided) 
 "input_example.csv"
 
## Requirements: 
- Python 3.10.2
- spaCy 3.7.2
- pandas 1.5.0
- numpy 1.23.3
