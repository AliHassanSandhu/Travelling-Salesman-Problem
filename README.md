# Problem
TSP

# Data
data.csv where each row represent differnt cities with coordinates x,y.

# Solution
Prepare data and create distance matrix (eculidean distance) so don't have to compute again for same cites.

Can represent cities as Index 0 for 1 city, 1 for 2nd and so on.

For 5 cities:
Chromosome can be represented as: [0,2,1,4,3,0] showing Salesman starting from 0th city and visiting every city onece and returning to the same city.

# To-Do's

Population Creation, Selection, Crossover, mutaion, 


# How to setup
pre-reqisuite: python 3.10+

### Clone this repo 
``` 
git clone <url>
cd <folder>
```
Create virtual Environment
```
python/python3 -m venv {env_name}
Example: python -m venv env
```
Activate Virtual Environment

Windows: 
``` 
{env_name}\Scripts\activate 
example: env\Script\activate
```
Linux/mac-os:
```
source env/bin/activate
```

Install Requirements

```
pip install -r requirements.txt
```

test the main file

```
python .\main.py
```