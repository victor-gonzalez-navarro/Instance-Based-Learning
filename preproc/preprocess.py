import numpy as np


class Preprocess:


    def preprocess_method(self, data, metric):
        features_del = []

        # Normalize the data as long as the metric distance is not HVDM
        if metric != 'hvdm':
            for feature in range(data.shape[1]):

                # Numerical Features
                if type(data[0, feature]) in [float, np.float64]:

                    # Calculate the mean of this feature of feed NaNs with it
                    mean_v = np.nanmean(data[:, feature], dtype=float)

                    # Calculate the max and min to normalize numerical data between 0 and 1
                    max_v = np.nanmax(data[:, feature])
                    min_v = np.nanmin(data[:, feature])

                    for sample in range(data.shape[0]):
                        if np.isnan(data[sample, feature]):
                            data[sample, feature] = mean_v
                        if max_v != 0:
                            data[sample, feature] = (data[sample, feature] - min_v) / (max_v - min_v)

            return data

        # In this case the data is not normalized, it is divided by 4*standard deviation
        else:
            for feature in range(data.shape[1]):

                # Numerical Features
                if type(data[0, feature]) in [float, np.float64]:

                    # Calculate the mean of this feature without considering NaNs
                    mean_v = np.nanmean(data[:, feature], dtype=float)

                    # Calculate the standard deviation of this feature without considering NaNs
                    std = np.nanstd(data[:, feature], dtype=float)

                    for sample in range(data.shape[0]):
                        if np.isnan(data[sample, feature]):
                            data[sample, feature] = mean_v/(4*std)
                        else:
                            data[sample, feature] = (data[sample, feature])/(4*std)

            return data
