# Automatic localisation by visual map
## Description
This code is a school project, where the goal was to estimate the most similar image to the input image from a sequence of images.

## Example
- The sequence of images is **_legumesA-XX_** where _XX_ is the image index in the sequence (00->09)

- The input image in the example is **_legumesB-00_**

- The model should predict the image **_legumesA-00_** as the most similar image to our example image input **_legumesB-00_**

## Approach
The idea was to use different algorithms (SIFT,SURF,KAZE,A-KAZE,ORB,BRISK and HoG) to extract local or global descriptors for each image, to get a visual map. 

Then algorithms like BF Matcher, FLANN, KNN, NNDR, Euclideean/Manhattan distance are used to match features vector.

This gives a probability distribution for each possible index. The predicted index for input image is the one with the largest probability.

## Model performance
Models are tested on 7 different sets of 2 sequences (A and B) in both directions (A->B and B->A).

