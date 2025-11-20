# Usage

1. Create a virtual environment, activate it, and install all the dependencies

```sh
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

2. Import the necessary functions/moduels from `psrpop` & synthesize a new population:

```python
from psrpop import *

pulsars = Simulate_Evolution(
    birth_rate=2 / 100,
    total_time=1e8,
    nbraking=3,
    beta=0.05,
)
```

3. Read the observational catalog:

```python
catalog = Read_Catalogue()
```

4. Select only the detectable pulsars (based on geometric criterion & radio flux):

```python
detectable_pulsars = pulsars.select(Detectable_Geometric).select(Detectable_Radioflux)
```

5. Overplot both populations:

```python
ax = Plot_PPdot(detectable_pulsars, color=None, s=0.1, lw=0)

ax.scatter(
    catalog.P0.astype(float),
    catalog.P1.astype(float),
    s=1,
    color="gray",
    zorder=-1,
)
```