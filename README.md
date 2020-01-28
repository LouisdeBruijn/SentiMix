# SentiMix Shared Task

Providing a space for development of our scripts for the SentiMix 2020 shared task.

Find it here https://competitions.codalab.org/competitions/20654#learn_the_details-overview


## Data class

data_manager.py contains the model of our data structure.

To manipulate the data, you can simply use the Preprocessor class to modify the data. The whole work flow is as simple as:

```
data = Data("PATH_TO_FILE", format="json")
data = Preprocessor.balance_data(data)
```

