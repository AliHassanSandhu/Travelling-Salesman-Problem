# Problem
Traveling Salesperson Problem (TSP) / Asymmetric TSP (ATSP) using Genetic Algorithm.

# Solution
Prepare data and compute a distance matrix (Euclidean or Asymmetric with directional modifiers/missing paths) once at startup to avoid recomputing during evaluation.

Cities are indexed as `0, 1, 2, ..., N-1`.

For 5 cities:
Chromosome is represented as: `[0, 2, 1, 4, 3, 0]` showing the salesman starting from city `0`, visiting every city exactly once, and returning to city `0`.

# Features
- **Matrix Caching:** Distance matrix computed once before GA execution.
- **Asymmetric Support:** Directional costs ($i \to j \neq j \to i$) and disconnected routes (`np.inf`).
- **Graph Visualization:** Uses NetworkX and Matplotlib to plot clear directed routes with curved arrows and weight labels.

# To-Do's
- [x] Data Preparation & Distance Matrix Generation
- [x] Initial Population Creation
- [x] Fitness Evaluation (handling `np.inf` paths)
- [x] Parent Selection (Tournament / Roulette Wheel)
- [ ] Crossover Operator (Ordered Crossover / OX)
- [ ] Mutation Operator (Swap / Inversion)
- [ ] Main GA Loop & Plotting Best Route

# How to setup
pre-requisite: python 3.10+

### Clone this repo 
```bash
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