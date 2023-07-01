
import math
import numpy as np

# Constants
T = 0.0125;        # Duration of heart beat (minutes)
TS = 0.0050;       # Duration of systole (minutes)
RS = 17.86;        # Resistance of systemic circulation (mmHg/(liter/minute))
CSA = 0.00175;     # Capacitance of systemic arteries (liters/mmHg)
PLA = 5.0;         # Left atrial pressure (mmHg)
RMi = 0.01;        # Mitral valve resistance (mmHg/(liter/minute))
RAo = 0.01;        # Aortic valve resistance (mmHg/(liter/minute))
CLVS = 0.00003;    # Compliance of left ventricle in systole (liters/mmHg)
CLVD = 0.0146;     # Compliance of left ventricle in diastole (liters/mmHg)
TAU_S = 0.0025;    # Time constant for ventricle pressure decay during systole (minutes)
TAU_D = 0.0075;    # Time constant for ventricle pressure decay during diastole (minutes)
CHECK = False;     # Set to True to turn on consistency checks


def CLV_func(t_in):
    t = t_in - np.floor(t_in / T) * T
    if t < TS:
        return CLVD * np.power(CLVS/CLVD, (1.0-np.exp(-t/TAU_S)) / (1.0-np.exp(-TS/TAU_S)) )
    return CLVS * np.power(CLVD/CLVS, (1.0-np.exp(-(t-TS)/TAU_D)) / (1.0-np.exp(-(T-TS)/TAU_D)) )


# State is np.array with [t, PLV, Psa, Smi, SAo, CLV]

def new_pressures(state, dt):
    t, P_LV_old, P_sa_old, S_mi, S_ao, C_LV_old = state
    C_LV = CLV_func(t+dt)

    c11 = C_LV + dt*((S_mi/RMi)+(S_ao/RAo))
    c12 = c21 = -dt*(S_ao/RAo)
    c22 = CSA + dt*((S_ao/RAo)+(1/RS))

    b1 = C_LV_old*P_LV_old + dt*S_mi/RMi*PLA
    b2 = CSA*P_sa_old
    D = c11*c22 - c12*c21

    P_LV = (c22*b1 - c12*b2) / D
    P_sa = (c11*b2 - c21*b1) / D

    if (CHECK):
        LHS1 = (C_LV*P_LV - C_LV_old*P_LV_old) / dt
        RHS1 = S_mi/RMi*(PLA-P_LV) - S_ao/RMi*(P_LV-P_sa)
        CH1 = LHS1 - RHS1
        LHS2 = CSA*(P_sa - P_sa_old) / dt
        RHS2 = S_ao/RAo*(P_LV-P_sa) - P_sa/RS
        CH2 = LHS2 - RHS2

        print(f"{CH1=}\t{CH2=}")
    
    return P_LV, P_sa


def step(state, dt):
    while True:
        S_mi_note = state[3]
        S_ao_note = state[4]

        P_LV, P_sa = new_pressures(state, dt)

        S_mi = 0 if P_LV > PLA else 1
        S_ao = 0 if P_sa > P_LV else 1


        state[1] = P_LV
        state[2] = P_sa
        state[3] = S_mi
        state[4] = S_ao

        if S_mi_note == S_mi and S_ao_note == S_ao:
            break

    state[0] += dt
    #cstate[1] = P_LV
    #state[2] = P_sa
    #state[3] = S_mi
    #state[4] = S_ao
    state[5] = CLV_func(state[0])


def run_model():
    state = np.array([0, 5.0, 90.0, 0, 0, CLV_func(0)])
    dt = 0.001
    while state[0] < 0.3:
        step(state, dt)
        print(f"{state[0]},{state[1]},{state[2]},{state[3]},{state[4]},{state[5]}")


if __name__ == "__main__":
    run_model()
