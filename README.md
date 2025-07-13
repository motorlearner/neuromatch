## Neuromatch

To get the processed data in you google colab notebook, insert the following as the first cell.

```
!rm -rf getdata.py* sample_data data01_direction4priors.csv
!wget -q https://raw.githubusercontent.com/motorlearner/neuromatch/refs/heads/main/getdata.py
%run getdata.py
```