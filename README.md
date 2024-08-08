The spline-based model has a lot of inputs. Below is a list of how they affect the outputted graph:

**Model variables:**
These variables affect the output of the given graph

Ticker: the ticker name from Yahoo Finance
Taylor degree: the Taylor degree of each spline (make sure to reset data if changing it)
Num_of_nodes: the number of splines within the graph
Interval_length: the time interval each spline runs for (measured in days)]
Weights: KNN weights (affect the entropy of training data)
Moving average: Moving average applied to raw data + derivatives

**Parameter selection:**
These variables affect where the graph begins

Use_rand_params: If set to true, the model will use a random row of data from the training data
Use_row_num: If set to true, the model will use the directed row number
Use_init_params: Allows for custom initial params (must be similar to weights)

**Data preprocessing**
These variables affect how the data is processed

Ticker: adds and processes the given ticker
Taylor degree: determines the number of derivatives taken
Weights: KNN weights (affect the entropy of training data)
Moving average: Moving average applied to raw data + derivatives
Start date: The start date of the imported data from Yahoo Finance
End date: The end date of the imported data from Yahoo Finance 
