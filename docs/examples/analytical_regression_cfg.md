# Analytical Regression with CGFs

In this example we will solve the [Kepler problem](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion),
a known instance of analytical regression. In analytical regression, we want to find a
mathematical formula that best approximates some input function.
The caveat is that we want the formula to be analytically described.

In this case we want to find the formula that best approximates the distance to orbital period
ratio in planetary motion. Kepler observed that for any planet in the solar system,
the ratio $R^3/T^2$ was a constant, where R is the radius of the orbit and T is the period.

The following table summarizes his observations.

```python
data = [
    # Planet     Radius      Period
    ("Mercury",  0.38710,    87.9693),
    ("Venus",    0.72333,    224.7008),
    ("Earth",    1,          365.2564),
    ("Mars",     1.52366,    686.9796),
    ("Jupiter",  5.20336,    4332.8201),
    ("Saturn",   9.53707,    10775.599),
    ("Uranus",   19.1913,    30687.153),
    ("Neptune",  30.0690,    60190.03),
]
```



```python
class Sum:
    def __init__(self, left, right):
        pass
```

