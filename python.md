python -m venv env
source env/bin/activate

>>> import os
>>> def ls(x):
...   print(os.listdir(x))
... 
>>> ls('generative_cfg')
>>> 
