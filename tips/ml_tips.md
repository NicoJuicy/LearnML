# Machine Learning Tips

## [What aspect ratio to use for line plots](https://towardsdatascience.com/should-you-care-about-the-aspect-ratio-when-creating-line-plots-ed423a5dceb3)

One of the most overlooked aspects of creating charts is the use of correct aspect ratios. 

### Calculating the aspect ratio

The concept of banking to 45 degrees is used to have coherency between the information presented and information perceived. 

Thus, we need to make sure that the orientation of the line segments in the chart is as close as possible to a slope of 45 degrees.

Here, the median absolute slope banking method has been used to calculate the aspect ratio for the sunspots plot. ￼

The ggthemes package provides a function called bank_slopes() to calculate the aspect ratio of the plot which takes x and y values as the two arguments. The default method is the median absolute slope banking. 

### Best practices:

- **Plotting multiple line graphs for comparison on a single chart:** The default aspect ratio works only if you do not plan to compare two different plots.


- **Comparing different line graphs from different charts:** Make sure the aspect ratio for each plot remains the same. Otherwise, the visual interpretation will be skewed. 

  1. Using incorrect or default aspect ratios: In this case, we choose the aspect ratios such that the plots end up being square-shaped.

  2. Calculating aspect ratios per plot: The best approach to compare the plots is to calculate the aspect ratios for each plot. 

**Time-series:** It is best to calculate the aspect ratio since some hidden information can be more pronounced when using the correct aspect ratio for the plot.


## References


