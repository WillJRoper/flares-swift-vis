from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value
import numpy as np
import astropy.units as u

ts = np.arange(0.015, cosmo.age(0).value, 0.01)

print(ts)

zs = []
for t in ts:
    zs.append(z_at_value(cosmo.age, (t * u.Gyr), zmax=127))

print(len(zs))
print(zs)