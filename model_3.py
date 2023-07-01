
import numpy as np
import matplotlib.pyplot as plt

# Create index names
iLV = 0
isa = 1
isv = 2
iRV = 3
ipa = 4
ipv = 5
N = 6

# Initialize parameters
T = 0.0125;             # Duration of heart beat (minutes)
TS = 0.0050;            # Duration of systole (minutes)
TAU_S = 0.0025;         # Time constant for ventricle pressure decay during systole (minutes)
TAU_D = 0.0075;         # Time constant for ventricle pressure decay during diastole (minutes)
DT = 0.01*T;            # Time step duration (minutes)
NSTEPS = int(10*T/DT);   # Number of time steps

RS = 17.50;             # Systemic resistance (mmHg/(liter/minute))
RP = 1.79;              # Pulmonary resistance (mmHg/(liter/minute))
RMI = 0.0100;           # Mitral valve resistance (mmHg/(liter/minute))
RAO = 0.0100;           # Aortic valve resistance (mmHg/(liter/minute))
RTR = 0.0100;           # Tricuspid valve resistance (mmHg/(liter/minute))
RPU = 0.0100;           # Pulmonic valve resistance (mmHg/(liter/minute))

CSA = 0.00175;          # Systemic arterial compliance (liters/(mmHg))
CPA = 0.00412;          # Pulmonic arterial compliance (liters/(mmHg))
CSV = 1.7500;           # Systemic venous compliance (liters/(mmHg))
CPV = 0.0800;           # Pulmonic venous compliance (liters/(mmHg))
CLVS = 0.00003;         # Min (systolic) value of ventricular compliance (liters/(mmHg))
CLVD = 0.0146;          # Max (diastolic) value of ventricular compliance (liters/(mmHg))
CRVS = 0.0002;          # Min (systolic) value of right ventricular compliance (liters/(mmHg))
CRVD = 0.0365;          # Max (diastolic) value of right ventricular compliance (liters/(mmHg))

VSAD = 0.8250;          # Systemic arterial volume when pressure is zero (liters)
VPAD = 0.382;           # Pulmonic arterial volume when pressure is zero (liters)
VLVD = 0.027;           # Left ventricular volume when pressure is zero (liters)
VRVD = 0.027;           # Right ventricular volume when pressure is zero (liters)


# Debugging flag
CHECK = True
PLOT = True

# ind dict
ind_dict = {
    "iLV": 0,
    "isa": 1,
    "isv": 2,
    "iRV": 3,
    "ipa": 4,
    "ipv": 5,
    0: "iLV",
    1: "isa",
    2: "isv",
    3: "iRV",
    4: "ipa",
    5: "ipv",
}


def CLV_func(t_in):
    t = t_in - np.floor(t_in / T) * T
    if t < TS:
        return CLVD * np.power(CLVS/CLVD, (1.0-np.exp(-t/TAU_S)) / (1.0-np.exp(-TS/TAU_S)) )
    return CLVS * np.power(CLVD/CLVS, (1.0-np.exp(-(t-TS)/TAU_D)) / (1.0-np.exp(-(T-TS)/TAU_D)) )


def CRV_func(t_in):
    t = t_in - np.floor(t_in / T) * T
    if t < TS:
        return CRVD * np.power(CRVS/CRVD, (1.0-np.exp(-t/TAU_S)) / (1.0-np.exp(-TS/TAU_S)) )
    return CRVS * np.power(CRVD/CRVS, (1.0-np.exp(-(t-TS)/TAU_D)) / (1.0-np.exp(-(T-TS)/TAU_D)) )


def set_C_vec(C_vec, t_in):
    C_vec[iLV] = CLV_func(t_in)
    C_vec[iRV] = CRV_func(t_in)
    return C_vec


def set_S_mat(S_mat, P_vec):
    #S_mat = (P_vec[:, None] > P_vec).astype(float)
    for i in range(N):
        for j in range(N):
            S_mat[i, j] = 1.0 if P_vec[i] > P_vec[j] else 0.0
    return S_mat


def construct_A_matrix(S_mat, G_mat, C_vec, dt):
    A = -dt*np.multiply(S_mat, G_mat)
    A += -dt*np.multiply(np.transpose(S_mat), np.transpose(G_mat))

    summed_diag = np.sum(A, axis=1) - np.diag(A)

    #np.fill_diagonal(A, C_vec - summed_diag)
    for i in range(N):
        A[i, i] = C_vec[i] - summed_diag[i]

    return A


def get_new_P_vec(S_mat, G_mat, C_vec, dt, old_C_vec, old_P_vec):
    A = construct_A_matrix(S_mat, G_mat, C_vec, dt)
    P_vec = np.dot(np.linalg.inv(A), np.multiply(old_C_vec, old_P_vec))
    return P_vec


def system_step(S_mat, G_mat, C_vec, P_vec, t, dt):
    old_C_vec = C_vec.copy()
    old_P_vec = P_vec.copy()
    t += dt
    C_vec = set_C_vec(C_vec, t)
 
    while True:
        s_note = S_mat
        P_vec = get_new_P_vec(S_mat, G_mat, C_vec, dt, old_C_vec, old_P_vec)
        S_mat = set_S_mat(S_mat, P_vec)
        if np.all(s_note == S_mat):
            break
    
    if CHECK:
        ch = np.zeros_like(P_vec)
        for i in range(N):
            ch[i] = -(C_vec[i] * P_vec[i] - old_C_vec[i] * old_P_vec[i]) / dt
            for j in range(N):
                ch[i] += (S_mat[j,i]*G_mat[j,i] + S_mat[i,j]*G_mat[i,j])*(P_vec[j] - P_vec[i])
        print("Check: ", ch)
    
    return S_mat, G_mat, C_vec, P_vec, t


if __name__ == "__main__":
    # Initialize state variables
    C = np.zeros(N)
    P = np.zeros(N)
    S = np.zeros((N, N))
    G = np.zeros((N, N))

    # Additional Vd vector
    Vd = np.zeros(N)
    Vd[iLV] = VLVD
    Vd[iRV] = VRVD
    Vd[isa] = VSAD
    Vd[ipa] = VPAD

    # Initialize
    C[iLV] = CLV_func(0.0)
    C[isa] = CSA
    C[isv] = CSV
    C[iRV] = CRV_func(0.0)
    C[ipa] = CPA
    C[ipv] = CPV

    P[iLV] = 5.0
    P[isa] = 80.0
    P[isv] = 2.0
    P[iRV] = 2.0
    P[ipa] = 8.0
    P[ipv] = 5.0

    G[iLV, isa] = 1.0 / RAO
    #G[isa, iLV] = 1.0 / RAO
    G[isa, isv] = 1.0 / RS
    G[isv, isa] = 1.0 / RS
    G[isv, iRV] = 1.0 / RTR
    #G[iRV, isv] = 1.0 / RTR
    G[iRV, ipa] = 1.0 / RPU
    #G[ipa, iRV] = 1.0 / RPU
    G[ipa, ipv] = 1.0 / RP
    G[ipv, ipa] = 1.0 / RP
    G[ipv, iLV] = 1.0 / RMI
    #G[iLV, ipv] = 1.0 / RMI

    S = set_S_mat(S, P)

    # Create records
    t_rec = np.zeros(NSTEPS)
    P_rec = np.zeros((N, NSTEPS))
    V_rec = np.zeros((N, NSTEPS))
    S_rec = np.zeros((N, N, NSTEPS))

    t = 0.0
    for i in range(NSTEPS):
        S, G, C, P, t = system_step(S, G, C, P, t, DT)
        t_rec[i] = t
        P_rec[:, i] = P
        V_rec[:, i] = Vd + np.multiply(C, P)
        S_rec[:, :, i] = S

    if PLOT:

        # Plotting
        fig, axs = plt.subplots(2, 2)

        P_show = np.zeros((N,1))
        P_show[:,0] = P
        im0 = axs[0, 0].imshow(P_show)
        axs[0, 0].set_title('P')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks(np.arange(N))
        axs[0, 0].set_xticklabels([])
        axs[0, 0].set_yticklabels([ind_dict[i] for i in np.arange(N)])
        fig.colorbar(im0, ax=axs[0, 0])

        C_show = np.zeros((N,1))
        C_show[:,0] = C
        im1 = axs[0, 1].imshow(C_show)
        axs[0, 1].set_title('C')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks(np.arange(N))
        axs[0, 1].set_xticklabels([])
        axs[0, 1].set_yticklabels([ind_dict[i] for i in np.arange(N)])
        fig.colorbar(im1, ax=axs[0, 1])

        im2 = axs[1, 0].imshow(S)
        axs[1, 0].set_title('S')
        axs[1, 0].set_xticks(np.arange(N))
        axs[1, 0].set_yticks(np.arange(N))
        axs[1, 0].set_xticklabels([ind_dict[i] for i in np.arange(N)])
        axs[1, 0].set_yticklabels([ind_dict[i] for i in np.arange(N)])
        fig.colorbar(im2, ax=axs[1, 0])

        masked_G = np.ma.masked_where(G == 0, G)
        im3 = axs[1, 1].imshow(masked_G)
        axs[1, 1].set_title('G')
        axs[1, 1].set_xticks(np.arange(N))
        axs[1, 1].set_yticks(np.arange(N))
        axs[1, 1].set_xticklabels([ind_dict[i] for i in np.arange(N)])
        axs[1, 1].set_yticklabels([ind_dict[i] for i in np.arange(N)])
        fig.colorbar(im3, ax=axs[1, 1])
        plt.show()

        # Plot records
        for i in range(N):
            plt.plot(t_rec, P_rec[i], label=ind_dict[i])
        plt.title('P')
        plt.legend()
        plt.show()

        for i in range(N):
            plt.plot(t_rec, V_rec[i], label=ind_dict[i])
        plt.title('V')
        plt.legend()
        plt.show()

        plt.plot(t_rec, S_rec[ind_dict["iRV"], ind_dict["isa"], :]+0.00, label="Sao")
        plt.plot(t_rec, S_rec[ind_dict["iLV"], ind_dict["ipa"], :]+0.01, label="Spu")
        plt.plot(t_rec, S_rec[ind_dict["ipv"], ind_dict["iLV"], :]+0.02, label="Smi")
        plt.plot(t_rec, S_rec[ind_dict["isv"], ind_dict["iRV"], :]+0.03, label="Str")
        plt.title('S')
        plt.legend()
        plt.show()
