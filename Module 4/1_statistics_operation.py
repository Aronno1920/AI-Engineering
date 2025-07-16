import statistics as st

score = [72, 88, 91, 65, 70, 85, 95, 83, 77]

s_mean = st.mean(score)
print("Mean of the score: ", s_mean)
print('-----------------------\n')

s_median = st.median(score)
print("Median of the score: ", s_median)
print('-----------------------\n')

s_variance = st.variance(score)
print("Variance of the score: ", s_variance)
print('-----------------------\n')

s_deviation = st.stdev(score)
print("Standard Deviation of the score: ", s_deviation)
print('-----------------------\n')