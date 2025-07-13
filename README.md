## Neuromatch

To get the processed data in you google colab notebook, insert the following as the first cell.

```
!rm -rf getdata.py* sample_data data01_direction4priors.csv
!wget -q https://raw.githubusercontent.com/motorlearner/neuromatch/refs/heads/main/getdata.py
%run getdata.py
```

You will have access to `df`, which is a pandas dataframe with the processed data (you may wish to run `df = df` to enable autocomplete).

The naming convention for variables (with few exceptions) is roughly

> `what_how`

where `what` tells you what is being described (e.g. `stim` for stimulus, `resp` for response, `err` for error) and `how` tells you further details. A full list of column names and descriptions is printed out as a side effect of the first cell.