import numpy as np
import pint


def fins_resistance():
    D = 15
    L = np.array((3.8, 1.2, 1.0, 1.2, 3.8, 6.1))
    h = np.array((1, 1, 2, 1, 1, 1.5))
    b = 30*np.ones_like(L)
    A = np.sum(L*b)
    A_base = np.pi * D * b[0] / 2

    ratio = 1 + (A / A_base)
    return ratio


def main():
    ureg = pint.UnitRegistry()

    V = 12 * ureg.volt
    I = 0.309 * ureg.ampere  # Nominal current (max continuous current)
    q = (I * V).to(ureg.watt)
    print(q)

    R_conduction_housing = 31 * ureg.kelvin / ureg.watt

    h = 20 * (ureg.watt / (ureg.meter ** 2 * ureg.kelvin))
    
    print(fins_resistance())

    rad = 12e-3 * ureg.meter / 2
    length = 28e-3 * ureg.meter
    A = 2 * np.pi * rad * length
    print(A)
    R_convection = 1 / (h * A)
    R_tot = R_convection + R_conduction_housing
    R_tot = R_conduction_housing
    print(f'r_convection = {R_convection}')

    dT = q * R_tot / 1.7
    print(dT)
    

    R_convectiov = 1 / (h * A)
    # Nuselt = h * x / k


if __name__ == '__main__':
    main()