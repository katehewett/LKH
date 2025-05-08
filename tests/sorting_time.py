import numpy as np

from datetime import datetime

def create_2d_array(data_1d, dates):
  """
  Converts a 1D array to a 2D array, grouping data by date into columns.

  Args:
      data_1d (list or numpy.ndarray): The 1D array of data.
      dates (list of datetime objects): The list of dates corresponding to each data point.

  Returns:
      numpy.ndarray: The 2D array, padded with None where needed.
  """
  if len(data_1d) != len(dates):
    raise ValueError("Length of data_1d and dates must be equal.")

  # Group data by date using a dictionary to store each date with their corresponding values.
  data_by_date = {}
  for i, date in enumerate(dates):
      if date not in data_by_date:
          data_by_date[date] = []
      data_by_date[date].append(data_1d[i])
  
  # Find the maximum number of entries for any date
  max_rows = max(len(values) for values in data_by_date.values())
    
  # Create an empty 2D array with None as the fill value
  num_columns = len(data_by_date)
  result_2d = np.empty((max_rows, num_columns), dtype=object)
  result_2d[:] = None
  
  # Insert the date-based data into the new 2D array
  col_index = 0
  for date, values in data_by_date.items():
    for row_index, value in enumerate(values):
        result_2d[row_index, col_index] = value
    col_index += 1

  return result_2d


# Sample Usage
data_1d = [10, 12, 15, 20, 22, 18, 25, 30, 28, 35]
dates = [
    datetime(2025, 5, 1),
    datetime(2025, 5, 1),
    datetime(2025, 5, 2),
    datetime(2025, 5, 2),
    datetime(2025, 5, 1),
    datetime(2025, 5, 3),
    datetime(2025, 5, 3),
     datetime(2025, 5, 4),
    datetime(2025, 5, 4),
     datetime(2025, 5, 4),
]

result_array = create_2d_array(data_1d, dates)
print(result_array)