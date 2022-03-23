import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
import math
import MOFiT2_project_md as tim
import csv
import os

ab = 0.05292
m = 0.067
h = 1.
au_divider = 27211.6
hw = 10/au_divider
w = hw/h


def count_net(L, a):
    net = []
    net.append([1000, 1000])  # random number which is wrong in this place, so it will be seen if it's used

    for j in np.arange(-L/2, L/2 + a, a):
        for i in np.arange(-L/2, L/2 + a, a):
            net.append([i, j])

    return net


def count_nlg(N):
    nlg = [[0 for j in range(5)] for i in range(pow(N - 1, 2) + 1)]

    for i in range(1, pow(N - 1, 2) + 1):

        nlg[i][1] = i + math.ceil(i / (N - 1)) - 1
        nlg[i][2] = i + 1 + math.ceil(i / (N - 1)) - 1
        nlg[i][3] = i + (N - 1) + math.ceil(((N - 1) + i) / (N - 1)) - 1
        nlg[i][4] = i + (N - 1) + 1 + math.ceil(((N - 1) + i) / (N - 1)) - 1

    return nlg


def count_psi(xn, yn):
    psi: float = math.exp((-m*w/(2*h))*(pow(xn, 2) + pow(yn, 2)))
    return psi


def count_x(x1, x2, e1):
    x = (1-e1)*x1/2 + (1+e1)*x2/2
    return x


def count_y(y1, y3, e2):
    y = (1-e2)*y1/2 + (1+e2)*y3/2
    return y


def f1(eps):
    return (1-eps)/2.


def f2(eps):
    return (1+eps)/2.


def count_g(e1, e2):

    g1 = f1(e1)*f1(e2)
    g2 = f2(e1)*f1(e2)
    g3 = f1(e1)*f2(e2)
    g4 = f2(e1)*f2(e2)

    return [0, g1, g2, g3, g4]


def dg1(j, xl, xn, delta):
    g1 = count_g(xl + delta, xn)
    g2 = count_g(xl - delta, xn)
    g1 = g1[j + 1]
    g2 = g2[j + 1]
    return (g1 - g2)/(2*delta)


def dg2(j, xl, xn, delta):
    g1 = count_g(xl, xn + delta)
    g2 = count_g(xl, xn - delta)
    g1 = g1[j + 1]
    g2 = g2[j + 1]
    return (g1 - g2)/(2*delta)


def count_full_psi(net, nlg, filename, N, tab_psi=()):
    rows = []

    if os.path.exists(filename + '.csv'):
        os.remove(filename + '.csv')

    if len(tab_psi) == 0:
        tab_psi = [count_psi(p[0], p[1]) for p in net]

    with open(filename, 'w') as csvfile:

        writer = csv.writer(csvfile)

        for k in range(1, pow(N - 1, 2) + 1):
            for e1 in np.arange(-1, 1, 0.1):
                x = count_x(net[nlg[k][1]][0], net[nlg[k][2]][0], e1)

                for e2 in np.arange(-1, 1, 0.1):
                    y = count_y(net[nlg[k][1]][1], net[nlg[k][3]][1], e2)

                    S = 0.
                    g_tab = count_g(e1, e2)

                    for i in range(1, 5):
                        n = nlg[k][i]
                        S += tab_psi[n] * g_tab[i]

                    rows.append([x, y, S])

        rows.sort()
        writer.writerows(rows)


def plot_file(filename, L, N, plot_name="", f_name=""):
    x_tab = []
    y_tab = []
    Psi = []

    with open(filename, 'r', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            x_tab.append(float(row[0]) * ab)
            y_tab.append(float(row[1]) * ab)
            Psi.append(float(row[2]) * ab)

    fig, ax = plt.subplots()

    im = ax.scatter(x_tab, y_tab, c=Psi, cmap="gnuplot")
    fig.colorbar(im)
    ax.set_xlim([-L * ab / 2, L * ab / 2])
    ax.set_ylim([-L * ab / 2, L * ab / 2])

    if plot_name != "":
        plt.title(plot_name)

    if f_name != "":
        plt.savefig(str(f_name) + ".png", format='png')
        return f_name

    else:
        name = "Plot_psi_N_" + str(N)
        plt.savefig(name + ".png", format='png')
        return name


def count_S_T(a):

    J = pow(a, 2)/4
    A = pow(h, 2)/(2*m)

    w_tab = [5/9, 8/9, 5/9]
    p_tab = [-math.sqrt(3/5), 0, math.sqrt(3/5)]

    n = 4
    S = [[0 for j in range(n)] for i in range(n)]
    T = [[0 for j in range(n)] for i in range(n)]

    delta = 0.01

    for i in range(len(S)):
        for j in range(len(S[i])):

            sij = 0
            tij = 0

            for l in range(3):
                for n in range(3):
                    g = count_g(p_tab[l], p_tab[n])
                    sij += w_tab[l]*w_tab[n]*g[j+1]*g[i+1]

                    dgj1 = dg1(j, p_tab[l], p_tab[n], delta)
                    dgj2 = dg2(j, p_tab[l], p_tab[n], delta)
                    dgi1 = dg1(i, p_tab[l], p_tab[n], delta)
                    dgi2 = dg2(i, p_tab[l], p_tab[n], delta)
                    tij += w_tab[l] * w_tab[n] * (dgj1 * dgi1 + dgj2 * dgi2)

            S[i][j] = J * sij
            T[i][j] = A * tij

    return S, T


def count_S_H(S, T, net, nlg, N, a):

    w_tab = [5/9, 8/9, 5/9]
    p_tab = [-math.sqrt(3/5), 0, math.sqrt(3/5)]

    n = 4

    B = (pow(a, 2)/4)*m*pow(w, 2)/2
    V = [[[0 for j in range(n)] for i in range(n)] for k in range(pow(N - 1, 2))]
    S_full = [[0 for i in range(pow(N, 2))] for j in range(pow(N, 2))]
    H = [[0 for i in range(pow(N, 2))] for j in range(pow(N, 2))]

    for k in range(len(V)):
        for i in range(n):
            for j in range(n):

                vkij = 0

                for l in range(3):
                    for o in range(3):
                        g = count_g(p_tab[l], p_tab[o])
                        x = count_x(net[nlg[k][1]][0], net[nlg[k][2]][0], p_tab[l])
                        y = count_x(net[nlg[k][1]][1], net[nlg[k][3]][1], p_tab[o])
                        vkij += w_tab[l] * w_tab[o] * (pow(x, 2) + pow(y, 2)) * g[j+1] * g[i+1]

                V[k][i][j] = B * vkij

                S_full[nlg[k+1][i+1] - 1][nlg[k+1][j+1] - 1] += S[i][j]
                H[nlg[k+1][i+1] - 1][nlg[k+1][j+1] - 1] += T[i][j] + V[k][i][j]

    return S_full, H


def boundary_condition(S, H, net, N, L):
    p = (L/N)/10

    for i in range(1, len(net), 1):

        x = net[i][0]
        y = net[i][1]

        if abs((abs(x) - (L/2))) < p or abs((abs(y) - (L/2))) < p:

            for j in range(len(S[i-1])):
                S[i-1][j] = 0
                S[j][i-1] = 0

                H[i-1][j] = 0
                H[j][i-1] = 0

            S[i-1][i-1] = 1
            H[i-1][i-1] = -1410

    return S, H


def tabToString(tab):

    string = ""

    for x in tab:
        string += str(round(x, 3)) + ", "

    return string


def main():

    L: float = 100/ab
    N0: int = 2
    N = 2 * N0 + 1
    a = L / (2 * N0)

    filename = "data.csv"
    results = []
    results_energy = []
    plots = []

    net = count_net(L, a)

    ######################### PART 1 - PUT DATA TO TABLE (CREATING FILE AT THE END OF THE SCRIPT) ##############################

    nlg = count_nlg(N)

    for i in range(1, pow(N - 1, 2) + 1):
        for j in range(1, 5, 1):
            results.append([i, j, nlg[i][j], round(ab * net[nlg[i][j]][0], 3), round(ab * net[nlg[i][j]][1], 3)])

    ######################### PART 2 - COUNT AND PSI, SAVE IN FILE AND PLOT #############################

    count_full_psi(net, nlg, filename, N)
    plot_name = plot_file(filename, L, N)
    plots.append(plot_name)

    N0: int = 10
    N = 2 * N0 + 1
    a = L / (2 * N0)

    net = count_net(L, a)
    nlg = count_nlg(N)
    count_full_psi(net, nlg, filename, N)
    plot_name = plot_file(filename, L, N)
    plots.append(plot_name)

    ######################## PART 3 - FIND 15 SMALLEST ENERGY ##########################

    L_tab = [100, 200, 400]
    N0_tab = [2, 6, 10]

    for L in L_tab:
        L = L / ab

        for N0 in N0_tab:

            N = 2 * N0 + 1
            a = L / (2 * N0)
            net = count_net(L, a)
            nlg = count_nlg(N)

            S, T = count_S_T(a)
            S, H = count_S_H(S, T, net, nlg, N, a)
            S, H = boundary_condition(S, H, net, N, L)

            E = lin.eigvalsh(H, S)
            E_pos = np.array([e for e in E if e >= 0])
            E_pos = np.resize(E_pos, 15)

            results_energy.append([L * ab, N, tabToString(E_pos * au_divider)])

    ##################### PART 4 - PLOT WAVE FUNCTION FOR 6 SMALLEST STATES #############

    L = L_tab[0]/ab
    N0 = N0_tab[2]

    N = 2 * N0 + 1
    a = L / (2 * N0)
    net = count_net(L, a)
    nlg = count_nlg(N)

    S, T = count_S_T(a)
    S, H = count_S_H(S, T, net, nlg, N, a)
    S, H = boundary_condition(S, H, nlg, N, L)

    E, vec = lin.eigh(H, S)

    vec_new = [vec[:, i] for i in range(len(E))]

    E_pos = []
    vec = []

    for i, e in enumerate(E):

        if len(E_pos) >= 6:
            break
        elif e > 0:
            E_pos.append(e)
            vec.append(vec_new[i])

    for n, c in enumerate(vec):

        c = c.tolist()
        c.insert(0, None)
        count_full_psi(net, nlg, filename, N, c)
        plot_name = "Psi: L = " + str(L * ab) + ", N = " + str(N0) + ", n = " + str(n)
        f_name = "Plot_n_" + str(n)
        plot_name = plot_file(filename, L, N, plot_name, f_name)
        plots.append(plot_name)

    names = ["Number of element:", "Local number of node:", "Global number of node:", "x:", "y:"]
    names_energy = ["L [nm]", "N", "E [meV]"]
    tim.md_file(file_tit="MOFiT2_project", data_names=names, data_results=results,
                plots=plots, data_names_energy=names_energy, data_results_energy=results_energy)


main()
