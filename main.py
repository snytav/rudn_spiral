import numpy as np

from matplotlib.animation import ArtistAnimation

import matplotlib

from matplotlib import pyplot as plt
from scipy.special import iv, ivp

# Константы
B_0 = 100
Bc = 0.5
C = 30
z0 = 0
k = 0.349
kc = C * k
beta = np.array([0, -0.575, 0, -0.000799, 0, -0.00000156])
Cm = B_0 * beta
w = 1000000
tau = 0.001
e = 1
c = 1
m = 1

# Сетка 3D
r_linspace = np.linspace(0, 1, 50)
z_linspace = np.linspace(0, 1, 50)
theta_linspace = np.linspace(0, 2 * np.pi, 60)
thetag_3d, rg_3d, zg_3d = np.meshgrid(theta_linspace, r_linspace, z_linspace)
X_3d, Y_3d, Z_3d = rg_3d * np.cos(thetag_3d), rg_3d * np.sin(thetag_3d), zg_3d

# Сетка 2D (поперечный срез)
tg_2dp, rg_2dp = np.meshgrid(theta_linspace, r_linspace)

# Сетка 2D (продольный срез)
zg_2d, rg_2d = np.meshgrid(z_linspace, r_linspace)


def draw_3d(values, title='Untitled', vmin=None, vmax=None, ax_3d=None, save=None):
    if ax_3d is None:
        fig = plt.figure(figsize=(10, 9))
        ax_3d = fig.add_subplot(projection='3d')

    # Меняем местами x и z, чтобы цилиндр лежал на боку
    p = ax_3d.scatter(Z_3d, Y_3d, X_3d, c=values, s=10, cmap='plasma')  # norm=matplotlib.colors.LogNorm()
    # ax_3d.set_title(title, fontsize=14)
    ax_3d.set_xlabel('z', fontsize=14)

    if save is not None:
        plt.savefig(save + '.png')

    return p


def draw_circle_slice(values, at_z, title='Untitled', colorbar_title='Variable', vmin=None, vmax=None, save=None):
    closest_index = -1
    closest_diff = 9999999
    closest_z = -1

    for i in range(len(z_linspace)):
        z = z_linspace[i]
        diff = np.abs(z - at_z)
        if diff < closest_diff:
            closest_diff = diff
            closest_index = i
            closest_z = z

    closest_z = round(closest_z, 4)

    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(polar=True)
    plt.grid(False)
    # ax.set_title('Поперечный срез цилиндра\n' + title + f'\nz = {closest_z}')
    p = ax.pcolor(tg_2dp, rg_2dp, values[:, :, closest_index], cmap='plasma', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(p, ax=ax)
    # cb.set_label(colorbar_title, fontsize=14)

    if save is not None:
        plt.savefig(save + '.png')

    return p


def draw_rectangle_slice(values, title='Untitled', colorbar_title='Variable', vmin=None, vmax=None, save=None):
    fig, ax = plt.subplots(figsize=(6, 7))
    p = ax.pcolor(zg_2d, rg_2d, values[:, 0, :], cmap='plasma', vmin=vmin, vmax=vmax)

    # ax.set_title('Продольный срез цилиндра\n' + title, fontsize=14)
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('r', fontsize=14)

    cb = fig.colorbar(p, ax=ax)
    cb.set_label(colorbar_title, fontsize=14)

    if save is not None:
        plt.savefig(save + '.png')

    return p


def compute_B(derivative_fc_r, derivative_fc_theta, derivative_fc_z, derivative_fs_r, derivative_fs_theta,
              derivative_fs_z):
    Br = derivative_fs_r + derivative_fc_r
    Btheta = derivative_fs_theta + derivative_fc_theta
    Bz = B_0 + derivative_fs_z + derivative_fc_z

    return Br, Btheta, Bz


def compute_V(Er, Etheta, Ez, Br, Btheta, Bz):
    Vtheta = np.zeros((50, 60, 50))
    Vz = np.zeros((50, 60, 50))
    Vr = np.zeros((50, 60, 50))

    # Vz[:,:,0] = 1

    timesteps = 5

    Vr_last = Vr
    Vtheta_last = Vtheta
    Vz_last = Vz

    for i in range(timesteps):
        Vr = Vr_last + tau * e / (m * c) * (Vtheta_last * Bz - Vz_last * Btheta) + tau * e / m * Er
        Vtheta = Vtheta_last + tau * e / (m * c) * (-Vr_last * Bz + Vz_last * Br) + tau * e / m * Etheta
        Vz = Vz_last + tau * e / (m * c) * (-Vr_last * Btheta + Vtheta_last * Br) + tau * e / m * Ez

        Vr_last = Vr
        Vtheta_last = Vtheta
        Vz_last = Vz

    return Vr_last, Vtheta_last, Vz_last


if __name__ == '__main__':
    zeros = np.zeros((50, 60, 50))

    plt.rcParams.update({'figure.max_open_warning': 0})

    # fc = Bc / kc * np.cos(kc * (zg_3d - z0)) * iv(0, kc * rg_3d)
    fs = np.zeros((50, 60, 50))

    for m in range(1, 6):
        fs += Cm[m] * np.sin(m * (thetag_3d - k * zg_3d)) * iv(m, m * k * rg_3d)

    derivative_fs_r = 0.0
    for m in range(1, 6):
        multiplier1 = Cm[m] * np.sin(m * (thetag_3d - k * zg_3d))
        multiplier2 = 0.0

        for i in range(10):
            denom = np.math.factorial(i) * np.math.factorial(i + m)
            multiplier2 += ((m * k / 2) ** (2 * i + m)) * (2 * i + m) * (rg_3d ** (2 * i + m - 1)) / denom

        derivative_fs_r += multiplier1 * multiplier2

    derivative_fs_theta = 0.0
    for m in range(1, 6):
        derivative_fs_theta = derivative_fs_theta + Cm[m] * np.cos(m * (thetag_3d - k * zg_3d)) * m * iv(m,
                                                                                                         m * k * rg_3d)

    derivative_fs_z = 0.0
    for m in range(1, 6):
        derivative_fs_z = derivative_fs_z - Cm[m] * m * k * np.cos(m * (thetag_3d - k * zg_3d)) * iv(m, m * k * rg_3d)

    derivative_fc_r = 0.0
    multiplier = 0.0
    for i in range(1, 11):
        v = (kc / 2) ** (2 * i) * 2 * i * rg_3d ** (2 * i - 1) / np.math.factorial(i) ** 2
        multiplier += v

    derivative_fc_r = Bc / kc * np.cos(kc * (zg_3d - z0)) * multiplier

    derivative_fc_theta = 0.0

    derivative_fc_z = 0.0
    for m in range(10):
        derivative_fc_z = derivative_fc_z - Bc * np.sin(kc * (zg_3d - z0)) * iv(0, kc * rg_3d)

    Ar = 0.0
    for m in range(1, 6):
        v1 = Cm[m]
        v2 = np.sin(m * (thetag_3d - k * zg_3d))
        v3 = k * rg_3d * iv(m, m * k * rg_3d)
        Ar = Ar + v1 * v2 * v3

    Az = 0.0
    for m in range(1, 6):
        v1 = Cm[m]
        v2 = np.cos(m * (thetag_3d - k * zg_3d))
        v3 = k * rg_3d * ivp(m, m * k * rg_3d)
        Az = Az + v1 * v2 * v3
    Az = -Az

    Atheta_1 = B_0 * rg_3d / 2

    v1 = -Bc / kc
    v2 = np.sin(kc * (zg_3d - z0))
    v3 = iv(1, kc * rg_3d)

    Atheta_2 = v1 * v2 * v3

    AthetaSum = Atheta_1 + Atheta_2
    

    ### Er
    derivative_Atheta_r = 0.0
    multiplier = 0.0
    for j in range(10):
        multiplier = multiplier + (kc / 2) ** (2 * j + 1) * (2 * j + 1) * rg_3d ** (2 * j) / (
                    np.math.factorial(j) * np.math.factorial(j + 1))
    derivative_Atheta_r = B_0 / 2 - Bc / kc * np.sin(kc * (zg_3d - z0)) * multiplier

    derivative_Az_r = 0.0
    for m in range(1, 6):
        multiplier = 0.0
        for i in range(10):
            denom = (np.math.factorial(i) * np.math.factorial(i + m))
            multiplier += ((2 * i + m) ** 2) / denom * (0.5 ** (2 * i + m)) * ((m * k * rg_3d) ** (2 * i + m - 1))
        derivative_Az_r = derivative_Az_r - Cm[m] * np.cos(m * (thetag_3d - k * zg_3d)) * k * multiplier

    ### Etheta
    derivative_Az_theta = 0.0
    for m in range(1, 6):
        derivative_Az_theta = derivative_Az_theta + Cm[m] * k * rg_3d * ivp(m, m * k * rg_3d) * np.sin(
            m * (thetag_3d - k * zg_3d)) * m

    ### Ez
    derivative_Az_z = 0.0
    for m in range(1, 6):
        derivative_Az_z = derivative_Az_z - Cm[m] * np.sin(m * (thetag_3d - k * zg_3d)) * m * k ** 2 * rg_3d * ivp(m,
                                                                                                                   m * k * rg_3d)
    derivative_Atheta_z = -Bc * np.cos(kc * (zg_3d - z0)) * iv(1, kc * rg_3d)

    phi_corrugation = w / k * (rg_3d * B_0 * rg_3d / 2 + Az / k)
    phi_spiral = w / k * (rg_3d * AthetaSum)
    phi_full = w / k * (rg_3d * AthetaSum + Az / k)

    # Er_full     = w / k * (AthetaSum + rg_3d * derivative_Atheta_r + 1 / k * derivative_Az_r)
    # Etheta_full = w / (k ** 2) * derivative_Az_theta
    # Ez_full     = w / k * (rg_3d * derivative_Atheta_z + 1 / k * derivative_Az_z)

    # Er_corrugation     = w / k * (B_0 * rg_3d + 1 / k * derivative_Az_r)
    # Etheta_corrugation = w / (k ** 2) * derivative_Az_theta
    # Ez_corrugation     = w / (k ** 2) * derivative_Az_z

    Er_spiral = w / k * (AthetaSum + rg_3d * derivative_Atheta_r)
    Etheta_spiral = np.zeros((50, 60, 50))
    Ez_spiral = w / k * rg_3d * derivative_Atheta_z

    Br_spiral, Btheta_spiral, Bz_spiral = compute_B(zeros, zeros, zeros, derivative_fs_r, derivative_fs_theta,
                                                    derivative_fs_z)
    # Br_corrugation, Btheta_corrugation, Bz_corrugation = compute_B(derivative_fc_r, derivative_fc_theta, derivative_fc_z, zeros, zeros, zeros)
    # Br_full, Btheta_full, Bz_full = compute_B(derivative_fc_r, derivative_fc_theta, derivative_fc_z, derivative_fs_r, derivative_fs_theta, derivative_fs_z)

    Vr_spiral, Vtheta_spiral, Vz_spiral = compute_V(Er_spiral, Etheta_spiral, Ez_spiral, Br_spiral, Btheta_spiral,
                                                    Bz_spiral)
    # Vr_corrugation, Vtheta_corrugation, Vz_corrugation = compute_V(Er_corrugation, Etheta_corrugation, Ez_corrugation, Br_corrugation, Btheta_corrugation, Bz_corrugation)
    # Vr_full, Vtheta_full, Vz_full = compute_V(Er_full, Etheta_full, Ez_full, Br_full, Btheta_full, Bz_full)

    component_title = ['r', 'theta', 'z']

    # Отрисовка E
    for i, E in enumerate([Er_spiral, Etheta_spiral, Ez_spiral]):
        title = component_title[i]
        bmin = E.min()
        bmax = E.max()
        draw_3d(E, vmin=bmin, vmax=bmax, title="E" + title + "spiral")
        draw_rectangle_slice(E, vmin=bmin, vmax=bmax, title="E" + title + "spiral")
        draw_circle_slice(E, 0.5, vmin=bmin, vmax=bmax)

    # Отрисовка B
    for i, B in enumerate([Br_spiral, Btheta_spiral, Bz_spiral]):
        title = component_title[i]
        bmin = B.min()
        bmax = B.max()
        draw_3d(B, vmin=bmin, vmax=bmax, save='B' + title + '_spiral_3d' + '_' + str(C))
        draw_rectangle_slice(B, vmin=bmin, vmax=bmax, save='B' + title + '_spiral_rect' + '_' + str(C))
        draw_circle_slice(B, 0.5, vmin=bmin, vmax=bmax, save='B' + title + '_spiral_circle' + '_' + str(C))

    # Отрисовка V
    for i, V in enumerate([Vr_spiral, Vtheta_spiral, Vz_spiral]):
        title = component_title[i]
        vmin = V.min()
        vmax = V.max()
        draw_3d(V, vmin=vmin, vmax=vmax, save='V' + title + '_spiral_3d' + '_' + str(C))
        draw_rectangle_slice(V, vmin=vmin, vmax=vmax, save='V' + title + '_spiral_rect' + '_' + str(C))
        draw_circle_slice(V, 0.5, vmin=vmin, vmax=vmax, save='V' + title + '_spiral_circle' + '_' + str(C))


