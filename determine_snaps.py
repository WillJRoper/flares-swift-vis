from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value
import numpy as np
import astropy.units as u

ts = np.arange(0.015, cosmo.age(0).value, 0.01)

print(ts)
print((0.01 * u.Gyr).to(u.Myr))

zs = []
for t in ts:
    zs.append(z_at_value(cosmo.age, (t * u.Gyr), zmax=127))
    print("%.16f" % zs[-1])
print(0.0)

zs.append(0.0)

print(len(zs))
# print(np.array(zs))
